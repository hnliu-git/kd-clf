import yaml
import pytorch_lightning as pl

from utils import serialize_config
from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from model.bert_base_kd import BertBaseKD
from pytorch_lightning.loggers import WandbLogger


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    parser = ClfDataModule.add_model_specific_args(parser)
    parser = BertBaseKD.add_model_specific_args(parser)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


if __name__ == '__main__':

    pl.seed_everything(2022)
    args = get_args('configs/bert_base_kd.yaml')
    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    dm = ClfDataModule(load_dataset('glue', 'sst2'), args)
    model = BertBaseKD(args)
    trainer = Trainer(gpus=1, logger=wandb_logger)

    trainer.fit(model, dm)


