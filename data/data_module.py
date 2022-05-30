'''
Data module for pretraining and classification task, require a dataset object as input
'''

import torch

from typing import Optional
from itertools import chain
from argparse import ArgumentParser
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class PtrDataModule(LightningDataModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--num_workers", type=int, default=1,
                            help="Number of workers for data loading.")
        parser.add_argument("--tokenizer", type=str, default="prajjwal1/bert-tiny",
                            help="tokenizer model")
        parser.add_argument("--max_length", type=int, default=512)
        parser.add_argument("--val_split_per", type=float, default=10,
                            help="Percentage of spliting if the dataset doesn't have validation key")
        parser.add_argument("--mlm_prob", type=float, default=0.15,
                            help="mlm probability")
        return parser

    def __init__(self, dataset, args):
        '''
        :param dataset:  A dataset object,
                         see https://huggingface.co/docs/datasets/access.html
        '''
        super(PtrDataModule, self).__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.raw_dataset = dataset

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=args.mlm_prob)

    def setup(self, stage: Optional[str] = None) -> None:

        def tokenize_function(examples):
            return self.tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        if not self.args.load_data_dir:
            max_seq_length = self.args.max_seq_length
            column_names = self.raw_dataset['train'].column_names
            text_column_name = "text" if "text" in column_names else column_names[0]
            tokenized_datasets = self.raw_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=40000,
                keep_in_memory=True,
                num_proc=self.args.num_workers,
                remove_columns=column_names,
                desc="Running tokenizer on every text in dataset"
            )
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                keep_in_memory=True,
                batch_size=40000,
                num_proc=self.args.num_workers,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )
            if self.args.save_data_dir:
                tokenized_datasets.save_to_disk(self.args.save_data_dir)
        else:
            tokenized_datasets = self.raw_dataset

        self.train = tokenized_datasets['train']
        self.val = tokenized_datasets['validation']

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collator
        )
        return self.val_loader


class ClfDataModule(LightningDataModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=256,
                            help="Batch size.")
        parser.add_argument('--max_length', type=int, default=128,
                            help='max sequence length')
        parser.add_argument("--num_workers", type=int, default=8,
                            help="Number of workers for data loading.")
        parser.add_argument("--tokenizer", type=str, default="prajjwal1/bert-tiny",
                            help="tokenizer model")

        return parser

    @staticmethod
    def collate_fn_def(batch):
        has_label = True if 'label' in batch[0] else False

        batch_dict = {
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'labels': []
        }

        if not has_label:
            batch_dict.pop('labels')

        for item in batch:
            batch_dict['input_ids'].append(torch.LongTensor(item['input_ids']))
            batch_dict['token_type_ids'].append(torch.LongTensor(item['token_type_ids']))
            batch_dict['attention_mask'].append(torch.LongTensor(item['attention_mask']))
            if has_label: batch_dict['labels'].append(torch.LongTensor([item['label']]))

        batch_dict = {k: torch.stack(v, dim=0) for k, v in batch_dict.items()}

        if has_label:
            batch_dict['labels'] = torch.squeeze(batch_dict['labels'])

        return batch_dict

    def __init__(self, dataset, hparams):
        '''
        :param dataset:  A dataset object containing keys ['train', 'validation, 'test'],
                         see https://huggingface.co/docs/datasets/access.html
        '''
        super(ClfDataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer)

    def setup(self, stage: Optional[str] = None) -> None:
        self.text_column = self.dataset.column_names['train'][0]

        self.train = self.dataset['train']
        self.val = self.dataset['validation']
        self.test = self.dataset['test']

        self.train = self.train.map(lambda e: self.tokenizer(e[self.text_column],
                                                                 truncation=True,
                                                                 padding='max_length',
                                                                 max_length=self.hparams.max_length))
        self.val = self.val.map(lambda e: self.tokenizer(e[self.text_column],
                                                                 truncation=True,
                                                                 padding='max_length',
                                                                 max_length=self.hparams.max_length))
        self.test = self.test.map(lambda e: self.tokenizer(e[self.text_column],
                                                                 truncation=True,
                                                                 padding='max_length',
                                                                 max_length=self.hparams.max_length))

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_def,
        )
        return self.train_loader

    def val_dataloader(self):
        self.valid_loader = DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_def,
        )
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_def,
        )
        return self.test_loader