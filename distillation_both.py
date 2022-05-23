import yaml
import pytorch_lightning as pl

from helper.adpator import *
from data.data_module import ClfDataModule
from helper.distiller import BaseDistiller

from utils import serialize_config, path_to_clf_model
from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Teacher Model
    parser.add_argument("--teacher_model", default='bert-base-uncased', type=str,
                        help="name of the teacher model")

    # Student Model
    parser.add_argument("--student_model", default=None, type=str,
                        help="pretrained student model, the default is the bert_tiny model")
    parser.add_argument("--ckpt_path", default='ckpts', type=str)

    # Data configs
    parser.add_argument("--trn_dataset", default=None, type=str, required=True)
    parser.add_argument("--val_dataset", default=None, type=str, required=True)

    # Adaptor configs
    parser.add_argument('--adaptors', default=[], type=list)

    parser = ClfDataModule.add_model_specific_args(parser)
    parser = BaseDistiller.add_model_specific_args(parser)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


def get_dataset_obj(args):
    sst2 = load_dataset('glue', 'sst2').rename_column('sentence', 'text')
    tweet = load_dataset('tweet_eval', 'sentiment')

    if args.trn_dataset == 'sst2':
        train = sst2['train']
    elif args.trn_dataset == 'tweet':
        train = tweet['train']
    elif args.trn_dataset == 'sst2-tweet':
        from datasets import concatenate_datasets
        train = concatenate_datasets([
            sst2.remove_columns(['label', 'idx'])['train'],
            tweet.remove_columns(['label'])['train']
        ])

    if args.val_dataset == 'sst2':
        args.num_classes = 2
        sst2['train'] = train
        dataset = sst2
    elif args.val_dataset == 'tweet':
        args.num_clases = 3
        tweet['train'] = train
        dataset = tweet

    args.num_training_steps = int(len(train)/args.batch_size) * args.epochs
    args.num_warmup_steps = int(len(train)/args.batch_size) * min(1, int(0.1*args.epochs))

    return dataset

from transformers.models.bert import modeling_bert, configuration_bert
from transformers import modeling_outputs


if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pl.seed_everything(2022)
    args = get_args('configs/distillation.yaml')

    # Logger
    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    # Data Module
    dm = ClfDataModule(get_dataset_obj(args), args)

    # Setup student and teacher
    teacher = path_to_clf_model(args.teacher_model, args.num_classes)
    student = path_to_clf_model(args.student_model, args.num_classes)

    str2adaptors = {
        'LogitMSE': LogitMSE(args.temperature),
        'LogitCE': LogitCE(args.temperature),
        'AttnTinyBERT': AttnTinyBERT(),
        'HidnTinyBERT': HidnTinyBERT(teacher.config.hidden_size, student.config.hidden_size),
        'EmbdTinyBERT': EmbdTinyBERT(teacher.config.hidden_size, student.config.hidden_size),
        'AttnMiniLM': AttnMiniLM(),
        'ValMiniLM': ValMiniLM(),
        'HidnPKD': HidnPKD(teacher.config.hidden_size, student.config.hidden_size),
    }

    # Setup adaptors
    adaptors = torch.nn.ModuleList([
       str2adaptors[name] for name in args.adaptors
    ])

    # Setup lightning
    distiller = BaseDistiller(
        teacher,
        student,
        args,
        adaptors,
    )

    trainer = Trainer(
        gpus=1,
        logger=wandb_logger,
        max_epochs=args.epochs,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(distiller, dm)
