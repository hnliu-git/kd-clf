import textbrewer
from textbrewer import GeneralDistiller
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from textbrewer import TrainingConfig, DistillationConfig
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AdamW
)

import yaml
import pytorch_lightning as pl

from helper.adpator import *
from data.data_module import ClfDataModule
from helper.distiller import BaseDistiller, HgCkptIO

from utils import serialize_config, path_to_clf_model
from datasets import load_dataset
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from datasets import load_dataset, load_metric

device = 'cuda:0'


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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = get_args('configs/distillation.yaml')

    # Logger
    # wandb_logger = WandbLogger(project=args.project, name=args.exp)

    # Data Module
    dm = ClfDataModule(get_dataset_obj(args), args)
    dm.setup()

    train_dataloader = dm.train_dataloader()

    teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    student_model = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2').to(device)

    teacher_model.config.output_values = True
    teacher_model.config.output_attentions = True
    teacher_model.config.output_hidden_states = True

    student_model.config.output_values = True
    student_model.config.output_attentions = True
    student_model.config.output_hidden_states = True

    # Show the statistics of model parameters
    print("\nteacher_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(teacher_model,max_level=3)
    print (result)

    print("student_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(student_model,max_level=3)
    print (result)

    num_epochs = 20
    num_training_steps = len(train_dataloader) * num_epochs
    # Optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=1e-4)

    scheduler_class = get_linear_schedule_with_warmup
    # arguments dict except 'optimizer'
    scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}

    def simple_adaptor(batch, model_outputs):
        return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}

    distill_config = DistillationConfig(
        intermediate_matches=[
         {'layer_T':0, 'layer_S':0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
         {'layer_T':8, 'layer_S':2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1}])
    train_config = TrainingConfig()

    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


    with distiller:
        distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=None)

