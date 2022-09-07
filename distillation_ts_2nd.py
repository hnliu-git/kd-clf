
import pytorch_lightning as pl

from data.data_module import ClfDataModule
from helper.distiller import *

from datasets import load_dataset
from pytorch_lightning import Trainer
from utils import get_distillation_args
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoModelForSequenceClassification


def get_model(name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.output_values = True

    return model


if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pl.seed_everything(2022)
    args = get_distillation_args('configs/distillation.yaml')

    teacher_model = 'xlm-roberta-base'
    student_model = 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'

    # Data Module
    dataset = load_dataset('tweet_eval', 'sentiment')
    num_training_steps = int(len(dataset['train']) / args.batch_size) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    dm = ClfDataModule(
        dataset,
        tokenizer=teacher_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Setup student and teacher
    teacher = get_model(teacher_model, 3)
    student = get_model(student_model, 3)

    distiller = BaseDistiller(
        teacher,
        student,
        args.adaptors,
        dm,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    logger = WandbLogger(project=args.project, name=args.exp)

    trainer = Trainer(
        gpus=1,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(distiller, dm)
    trainer.test(distiller, dm)
