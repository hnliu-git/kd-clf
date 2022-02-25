import yaml
import pytorch_lightning as pl

from utils import serialize_config
from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from pytorch_lightning.loggers import WandbLogger
from model.model_finetune import ClfFinetune, HgCkptIO
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    parser = ClfDataModule.add_model_specific_args(parser)
    parser = ClfFinetune.add_model_specific_args(parser)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


if __name__ == '__main__':

    pl.seed_everything(2022)
    args = get_args('configs/finetune.yaml')
    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    dm = ClfDataModule(load_dataset('glue', 'sst2'), args)
    model = ClfFinetune(args)

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        monitor='val_loss',
        save_on_train_epoch_end=True,
        filename="%s-{epoch:02d}-{val_loss:.2f}"
                 % (args.model.split('/')[-1]),
    )

    early_stopping = EarlyStopping(
        mode='min',
        patience=6,
        min_delta=0.01,
        monitor='val_loss'
    )

    trainer = Trainer(
        gpus=1,
        plugins=[HgCkptIO()],
        logger=wandb_logger,
        callbacks=[ckpt_callback, early_stopping]
    )

    trainer.fit(model, dm)

