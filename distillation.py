
import yaml
import pytorch_lightning as pl

from model.adpator import *
from data.data_module import ClfDataModule
from model.distiller import BaseDistiller

from utils import serialize_config
from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    AutoModelForSequenceClassification
)


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
    parser.add_argument("--hidden_size", default=768, type=int,
                        help="Dim of the encoder layer and pooler layer of the student")
    parser.add_argument("--hidden_layers", default=12, type=int,
                        help="Number of hidden layers in encoder of the student")
    parser.add_argument("--atten_heads", default=12, type=int,
                        help="Number of attention heads of the student")

    parser = ClfDataModule.add_model_specific_args(parser)
    parser = BaseDistiller.add_model_specific_args(parser)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


def get_teacher_and_student(args):
    if args.student_model:
        student = AutoModelForSequenceClassification.from_pretrained(args.student_model, num_labels=args.num_classes)
    else:
        config = BertConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.hidden_layers,
            num_attention_heads=args.atten_heads,
            num_labels=args.num_classes
        )
        student = BertForSequenceClassification(config)

    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, num_labels=args.num_classes)

    teacher.config.output_attentions = True
    teacher.config.output_hidden_states = True
    student.config.output_hidden_states = True
    student.config.output_attentions = True

    return teacher, student


if __name__ == '__main__':

    pl.seed_everything(2022)
    args = get_args('configs/distillation.yaml')

    # Logger
    wandb_logger = WandbLogger(project=args.project, name=args.exp)

    # Data Module
    if args.dataset_name == 'sst2':
        args.num_classes = 2
        dm = ClfDataModule(load_dataset('glue', 'sst2'), args)
    elif args.dataset_name == 'tweet':
        args.num_classes = 3
        dm = ClfDataModule(load_dataset('tweet_eval', 'sentiment'), args)

    # Setup student and teacher
    teacher, student = get_teacher_and_student(args)

    # Setup adaptor
    attn_adaptor = AttnMiniLMAdaptor()
    # w_hidn = torch.rand(teacher.config.hidden_size, student.config.hidden_size, requires_grad=True).cuda()
    hidn_adaptor = HidnRelnAdaptor()

    # Setup lightning
    distiller = BaseDistiller(
        teacher,
        student,
        args,
        attn_adaptor,
        hidn_adaptor
    )

    early_stopping = EarlyStopping(
        mode='min',
        patience=6,
        min_delta=0.01,
        monitor='val_nll_loss'
    )

    trainer = Trainer(
        # gpus=1,
        logger=wandb_logger,
        callbacks=[early_stopping]
    )

    trainer.fit(distiller, dm)


