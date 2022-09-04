import yaml
import pytorch_lightning as pl

from data.data_module import ClfDataModule
from helper.distiller import *

from utils import *
from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForSequenceClassification
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Data configs
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size.")
    parser.add_argument('--max_length', type=int, default=128,
                        help='max sequence length')
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading.")
    parser.add_argument("--tokenizer", type=str, default="prajjwal1/bert-tiny",
                        help="tokenizer model")

    # Distillation configs
    parser.add_argument('--adaptors', default=[], type=list)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--temperature", default=4, type=float)
    parser.add_argument("--ckpt_path", default='ckpts', type=str)

    # Optimizer configs
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


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
    args = get_args('configs/distillation.yaml')
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

    # Setup lightning
    distiller = BaseDistiller(
        teacher,
        student,
        args.adaptors,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    logger = WandbLogger(project=args.project, name=args.exp)

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        monitor='attentions:mse',
        save_last=True,
        filename="1st-%s"
                 % (student_model.split('/')[-1]),
    )

    trainer = Trainer(
        # gpus=1,
        logger=logger,
        plugins=[HgCkptIO()],
        max_epochs=args.epochs,
        callbacks=[
            ckpt_callback,
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(distiller, dm)
    trainer.test(distiller, dm)
