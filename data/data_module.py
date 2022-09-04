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


class ClfDataModule(LightningDataModule):

    @staticmethod
    def collate_fn_def(batch):

        keys = list(batch[0].keys())
        keys.remove('text')
        if 'label' in keys:
            keys.remove('label')
            keys.append('labels')

        batch_dict = {key: [] for key in keys}

        for item in batch:
            for key in keys:
                if key == 'labels':
                    batch_dict[key].append(torch.LongTensor([item['label']]))
                else:
                    batch_dict[key].append(torch.LongTensor(item[key]))

        batch_dict = {k: torch.stack(v, dim=0) for k, v in batch_dict.items()}

        if 'labels' in batch_dict:
            batch_dict['labels'] = torch.squeeze(batch_dict['labels'])

        return batch_dict

    def __init__(self, dataset, tokenizer,
                 max_length=128, batch_size=32,
                 num_workers=4):
        '''
        :param dataset:  A dataset object containing keys ['train', 'validation, 'test'],
                         see https://huggingface.co/docs/datasets/access.html
        '''
        super(ClfDataModule, self).__init__()

        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def setup(self, stage: Optional[str] = None) -> None:
        train_col_names = self.dataset['train'].column_names
        self.text_col = 'text' if 'text' in train_col_names else train_col_names[0]

        self.train = self.dataset['train']
        self.val = self.dataset['validation']
        self.test = self.dataset['test']

        self.train = self.train.map(lambda e: self.tokenizer(e[self.text_col],
                                                                 truncation=True,
                                                                 padding='max_length',
                                                                 max_length=self.max_length),
                                                                 num_proc=4)
        self.val = self.val.map(lambda e: self.tokenizer(e[self.text_col],
                                                                 truncation=True,
                                                                 padding='max_length',
                                                                 max_length=self.max_length),
                                                                 num_proc=4)
        self.test = self.test.map(lambda e: self.tokenizer(e[self.text_col],
                                                                 truncation=True,
                                                                 padding='max_length',
                                                                 max_length=self.max_length),
                                                                 num_proc=4)

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_def,
        )
        return self.train_loader

    def val_dataloader(self):
        self.valid_loader = DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_def,
        )
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_def,
        )
        return self.test_loader