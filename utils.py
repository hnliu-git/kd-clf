import yaml
import datasets

from typing import List, Dict
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification
)


def get_clf_dataset_obj(args):
    sst2 = datasets.load_from_disk('data/sst2')
    tweet = datasets.load_from_disk('data/tweet')

    if args.trn_dataset == 'sst2':
        train = sst2['train']
    elif args.trn_dataset == 'tweet':
        train = tweet['train']
    elif args.trn_dataset == 'sst2-tweet':
        train = datasets.concatenate_datasets([
            sst2.remove_columns(['label', 'idx'])['train'],
            tweet.remove_columns(['label'])['train']
        ])
    else:
        train = datasets.load_dataset(args.trn_dataset)['train']
        train.remove_columns(['label'])

    if args.val_dataset == 'sst2':
        args.num_classes = 2
        sst2['train'] = train
        dataset = sst2
    elif args.val_dataset == 'tweet':
        args.num_classes = 3
        tweet['train'] = train
        dataset = tweet

    args.num_training_steps = int(len(train)/args.batch_size) * args.epochs
    args.num_warmup_steps = int(len(train)/args.batch_size) * min(1, int(0.1*args.epochs))

    return dataset


def path_to_clf_model(path: str, num_classes: int):
    if 'yaml' in path:
        args = yaml.load(open(path), Loader=yaml.FullLoader)
        config = BertConfig(**args, num_labels=num_classes)
        model = BertForSequenceClassification(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_classes)

    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.output_values = True

    return model


def path_to_mlm_model(path: str):
    if 'yaml' in path:
        args = yaml.load(open(path), Loader=yaml.FullLoader)
        config = BertConfig(**args)
        model = BertForMaskedLM(config)
    else:
        model = BertForMaskedLM.from_pretrained(path)

    return model


def serialize_config(config: Dict) -> List[str]:
    """"""

    def parse_value(value):
        if isinstance(value, int) or isinstance(value, float) or \
           isinstance(value, str) or isinstance(value, bool):
            return str(value)
        elif isinstance(value, List):
            return [str(val) for val in value]
        else:
            raise ValueError(f"Invalid value in config file: {value}")

    # Get an empty list for serialized config:
    serialized_config = []

    for key, value in config.items():
        # Append key:
        serialized_config.append("--" + key)

        # Append value:
        if isinstance(value, bool):
            continue
        elif isinstance(value, Dict):
            serialized_config.pop()
            for k, v in value.items():
                serialized_config.append("--" + k)
                if isinstance(v, bool): continue
                serialized_config.append(parse_value(v))
        else:
            serialized_config.append(parse_value(value))

    return serialized_config