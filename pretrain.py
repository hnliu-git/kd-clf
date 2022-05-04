import transformers
import yaml

from utils import *

from datasets import load_dataset, load_from_disk
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from helper.pretrainer import Pretrainer, HgCkptIO
from data.data_module import PtrDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Dataset
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_cfgs', type=str, default=None)
    parser.add_argument("--load_data_dir", type=str, default=None)
    parser.add_argument('--save_data_dir', type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Student Config
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model', default=None, required=True, type=str,
                        help="path to the configurations of the model")

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

    if args.load_data_dir:
        dataset = load_from_disk(args.load_data_dir)
    else:
        dataset = prepare_dataset(args.dataset_name, args.dataset_cfgs, args)

    args.num_training_steps = int(len(dataset['train'])/args.batch_size) * args.epochs
    args.num_warmup_steps = int(0.01 * args.num_training_steps)

    dm = PtrDataModule(dataset, args)
    model = path_to_mlm_model(args.model)
    pretrainer = Pretrainer(model, args)

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        every_n_epochs=10,
        monitor='perplexity',
        mode='min',
        filename="%s-{epoch:02d}-{perplexity:.2f}"
                 % (args.model.split('/')[-1]),
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=[args.gpu],
        logger=wandb_logger,
        plugins=[HgCkptIO()],
        max_epochs=args.epochs,
        callbacks=[
            ckpt_callback,
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(pretrainer, dm)

    # fill = pipeline('fill-mask', model='filiberto', tokenizer='filiberto')
    # fill(f'ciao {fill.tokenizer.mask_token} va?')