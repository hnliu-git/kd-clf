import yaml
from datasets import load_dataset
import pytorch_lightning as pl

from utils import *
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from pytorch_lightning.loggers import WandbLogger
from helper.finetuner import ClfFinetune, HgCkptIO
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSequenceClassification


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Dataset
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size.")
    parser.add_argument('--max_length', type=int, default=128,
                        help='max sequence length')
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading.")

    # Model
    parser.add_argument("--ckpt_path", default='ckpts', type=str)

    # Training configs
    parser.add_argument("--epochs", default=5, type=int)

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

    pl.seed_everything(2022)
    args = get_args('configs/finetune.yaml')

    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    model_name = 'bert-base-uncased'
    dataset = load_dataset('tweet_eval', 'sentiment')
    model = get_model(model_name, 3)

    num_training_steps = int(len(dataset['train']) / args.batch_size) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    dm = ClfDataModule(
        dataset,
        tokenizer=model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    fintuner = ClfFinetune(
        model,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
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


