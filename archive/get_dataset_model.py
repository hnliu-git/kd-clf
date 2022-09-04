import datasets
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification
)


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


def get_amazon_dataset(args):
    dataset = datasets.load_from_disk('data/amazon_multi')
    args.num_classes = 3
    args.num_training_steps = int(len(dataset['train'])/args.batch_size) * args.epochs
    args.num_warmup_steps = int(len(dataset['train'])/args.batch_size) * min(1, int(0.1*args.epochs))

    return dataset

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
        train = datasets.concatenate_datasets([
            datasets.load_dataset(args.trn_dataset)['train'].rename_column('content', 'text').remove_columns(['label']),
            eval(args.val_dataset)['train'].remove_columns('label')
        ])

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