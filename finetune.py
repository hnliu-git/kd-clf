import yaml
import pytorch_lightning as pl

from utils import serialize_config, path_to_clf_model
from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from pytorch_lightning.loggers import WandbLogger
from helper.finetuner import ClfFinetune, HgCkptIO
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Dataset
    parser.add_argument('--dataset_name', type=str)

    # Model
    parser.add_argument("--model", default=None, required=True, type=str,
                        help="name of the model")

    parser = ClfDataModule.add_model_specific_args(parser)
    parser = ClfFinetune.add_model_specific_args(parser)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


def get_dataset_obj(args):

    if args.dataset_name == 'sst2':
        args.num_classes = 2
        dataset = load_dataset('glue', 'sst2')
    elif args.dataset_name == 'tweet':
        args.num_classes = 3
        dataset = load_dataset('tweet_eval', 'sentiment')

    args.num_training_steps = int(len(dataset['train'])/args.batch_size) * args.epochs
    args.num_warmup_steps = int(len(dataset['train'])/args.batch_size) * min(1, int(0.1*args.epochs))

    return dataset


if __name__ == '__main__':

    pl.seed_everything(2022)
    args = get_args('configs/finetune.yaml')

    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    dm = ClfDataModule(get_dataset_obj(args), args)
    model = path_to_clf_model(args.model, args.num_classes)
    fintuner = ClfFinetune(model, args)

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        monitor='val_loss',
        mode='min',
        filename="%s-{epoch:02d}-{val_loss:.2f}"
                 % (args.model.split('/')[-1]),
    )

    trainer = Trainer(
        gpus=1,
        # plugins=[HgCkptIO()],
        max_epochs=args.epochs,
        logger=wandb_logger,
        # callbacks=[ckpt_callback]
    )

    trainer.fit(fintuner, dm)


