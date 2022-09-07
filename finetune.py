
import pytorch_lightning as pl

from datasets import load_dataset
from utils import get_finetune_args
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from pytorch_lightning.loggers import WandbLogger
from helper.finetuner import ClfFinetune, HgCkptIO
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSequenceClassification


def get_model(name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.output_values = True

    return model


if __name__ == '__main__':

    pl.seed_everything(2022)
    args = get_finetune_args('configs/finetune.yaml')

    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    model_name = 'bert-base-uncased'

    dataset = load_dataset('tweet_eval', 'sentiment')
    model = get_model(model_name, 3)

    dm = ClfDataModule(
        dataset,
        tokenizer=model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers
    )

    fintuner = ClfFinetune(
        model, dm,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        monitor='val_loss',
        mode='min',
        filename="%s-{epoch:02d}-{val_loss:.2f}"
                 % (model_name.split('/')[-1]),
    )

    trainer = Trainer(
        # gpus=1,
        plugins=[HgCkptIO()],
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[ckpt_callback]
    )

    trainer.fit(fintuner, dm)


