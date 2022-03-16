import transformers
import yaml

from utils import *

from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from model.pretrainer import Pretrainer
from data.data_module import PtrDataModule
from pytorch_lightning.loggers import WandbLogger


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    parser = Pretrainer.add_model_specific_args(parser)
    parser = PtrDataModule.add_model_specific_args(parser)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


def prepare_dataset(dataset_name, dataset_cfg, args):

    raw_dataset = load_dataset(
        dataset_name,
        dataset_cfg,
        cache_dir=args.cache_dir
    )

    if "validation" not in raw_dataset.keys():
        raw_dataset['validation'] = load_dataset(
            dataset_name,
            dataset_cfg,
            split=f"train[:{args.val_split_per}%]",
            cache_dir=args.cache_dir
        )
        raw_dataset['train'] = load_dataset(
            dataset_name,
            dataset_cfg,
            split=f"train[{args.val_split_per}:]",
            cache_dir=args.cache_dir
        )

    return raw_dataset


if __name__ == '__main__':

    args = get_args('configs/pretrain.yaml')

    # Logger
    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    dataset = prepare_dataset('bookcorpus', None, args)
    dm = PtrDataModule(dataset, args)

    pretrainer = Pretrainer(args)
    trainer = Trainer(
        gpus=1,
        logger=wandb_logger
    )

    trainer.fit(pretrainer, dm)

    # fill = pipeline('fill-mask', model='filiberto', tokenizer='filiberto')
    # fill(f'ciao {fill.tokenizer.mask_token} va?')