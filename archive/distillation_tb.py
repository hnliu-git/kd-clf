import textbrewer
import pytorch_lightning as pl
from textbrewer import GeneralDistiller
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from textbrewer import TrainingConfig, DistillationConfig
from transformers import (
    AdamW
)

import yaml
from data.data_module import ClfDataModule
from utils import serialize_config, path_to_clf_model
from argparse import ArgumentParser

from datasets import load_dataset, load_metric

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

    # Training configs
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--learning_rate", type=float)

    parser = ClfDataModule.add_model_specific_args(parser)

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
    device = 'cuda:0'

    pl.seed_everything(2022)
    args = get_args('distillation_tb.yaml')

    # Data Module
    dm = ClfDataModule(get_dataset_obj(args), args)
    dm.setup()
    train_dataloader = dm.train_dataloader()

    teacher = path_to_clf_model(args.teacher_model, args.num_classes).to(device)
    student = path_to_clf_model(args.student_model, args.num_classes).to(device)

    # Show the statistics of model parameters
    print("\nteacher_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(teacher,max_level=3)
    print (result)

    print("student_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(student,max_level=3)
    print (result)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(student.parameters(), lr=args.learning_rate)
    # Scheduler
    scheduler_class = get_linear_schedule_with_warmup
    # arguments dict except 'optimizer'
    scheduler_args = {'num_warmup_steps':int(0.1*args.num_training_steps), 'num_training_steps':args.num_training_steps}

    def adaptor_T(batch, model_outputs):
        return {
            'logits': model_outputs.logits,
            'attention': model_outputs.attentions,
            'inputs_mask': batch['attention_mask']
        }

    def adaptor_S(batch, model_outputs):
        return {
            'logits': model_outputs.logits,
            'attention': model_outputs.attentions,
            'losses': model_outputs.loss,
            'inputs_mask': batch['attention_mask']
        }

    """
    Distillation Config
    ======================================================================
                Loss = w1L_kd + w2L_hl + w3sum(L_inter)
    ======================================================================
    'kd_loss_type': L_kd for 'logit' term from the adaptor, can be 'ce' or 'mse'
    'kd_loss_weight': w1 for L_kd
    'hard_label_weight': w2 for L_hl
    'kd_loss_weight_scheduler': scheduler for w1, can be 'linear_decay' or 'linear_growth'
    'hard_label_weight_scheduler': scheduler for w2, can be 'linear_decay' or 'linear_growth'
    'intermediate_matches' (List[Dict])
        'layer_T': T-th layer of teacher
        'layer_S': S-th layer of student
        'loss': can be
            - 'attention_mse', 'attention_mse_sum'
            - 'attention_ce', 'attention_ce_mean'
            - 'hidden_mse'
            - 'cos'
            - 'pkd'
            - 'nst': minilm relation hidden loss 
            - 'fsp'
        'weight': weight for the loss
        'proj': 
            - mapping func: can be 'linear', 'relu' (non-linear layer), 'tanh' (non-linear layer)
            - student dim
            - teacher dim
    """

    distill_config = DistillationConfig(
        kd_loss_type='mse',
        hard_label_weight=1,
        intermediate_matches=[
            {'layer_T': 11, 'layer_S': 3, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 1}
        ]
    )

    train_config = TrainingConfig(
        log_dir='./logs'
    )

    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher, model_S=student,
        adaptor_T=adaptor_T, adaptor_S=adaptor_S)

    with distiller:
        distiller.train(
            optimizer,
            train_dataloader,
            args.epochs,
            scheduler_class=scheduler_class,
            scheduler_args=scheduler_args,
            callback=None
        )

