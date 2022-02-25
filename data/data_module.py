'''
Data module for classification task, require a dataset object as input
Author: Haonan Liu
Last Modified: 09.02.2022
'''


import torch

from typing import Optional
from functools import partial
from argparse import ArgumentParser
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class ClfDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        '''
        :param data:  datasets.arrow_dataset.Dataset having keys ['sentence', 'label', 'idx']
        '''
        super(ClfDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ClfDataModule(LightningDataModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--num_workers", type=int, default=8,
                            help="Number of workers for data loading.")
        parser.add_argument("--tokenizer", type=str, default="prajjwal1/bert-tiny",
                            help="tokenizer model")
        return parser

    @staticmethod
    def default_collate_fn(batch, tkr):
        labels = []
        sents = []
        for item in batch:
            labels.append(item['label'])
            sents.append(item['sentence'])

        return {
            'sentence': tkr(sents, padding=True, return_tensors='pt'),
            'label': torch.LongTensor(labels)
        }

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
        self.train = ClfDataset(self.dataset['train'])
        self.val = ClfDataset(self.dataset['validation'])
        self.test = ClfDataset(self.dataset['test'])

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=partial(self.default_collate_fn, tkr=self.tokenizer),
        )
        return self.train_loader

    def val_dataloader(self):
        self.valid_loader = DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=partial(self.default_collate_fn, tkr=self.tokenizer),
        )
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=partial(self.default_collate_fn, tkr=self.tokenizer),
        )
        return self.test_loader