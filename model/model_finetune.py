from argparse import ArgumentParser
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import _PATH

import os
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities.cloud_io import get_filesystem
from transformers import (
    AutoModelForSequenceClassification,
)


class HgCkptIO(CheckpointIO):

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        '''Save the fine-tuned model in a hugging-face style.

        Args:
            checkpoint: ckpt, but only key 'hg_model' matters
            path: path to save the ckpt
            storage_options: not used
        '''
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint['hg_model'].save_pretrained(path.replace('.ckpt', ''))

    def load_checkpoint(self, path: _PATH, storage_options: Optional[Any] = None) -> Dict[str, Any]:
        pass

    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint
        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)


class ClfFinetune(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--ckpt_path", default='ckpts', type=str)
        parser.add_argument("--model", default='bert-base-uncased', type=str,
                            help="name of the model")
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--eps", default=1e-8, type=float)
        parser.add_argument("--num_classes", default=2, type=int)

        return parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model)

        # Metrics
        self.acc = torchmetrics.Accuracy(num_classes=self.hparams.num_classes)
        self.f1 = torchmetrics.F1Score(num_classes=self.hparams.num_classes)

    def forward(self, batch):
        """
        :param
            batch: {
                    sentence: dict from tokenizers
                    label: Tensor [bsz]
                   }
        :return:
            out: SequenceClassfierOutput with keys [loss, logits]
        """
        x, labels = batch['sentence'], batch['label']
        out = self.model(**x, labels=labels)
        return out

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=self.hparams.learning_rate,
                                      eps=self.hparams.eps,)

        fn_lambda = lambda epoch: 0.85 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [fn_lambda, fn_lambda])
        return [optimizer], [scheduler]

    def training_step(self, batch, idx):
        loss = self(batch).loss
        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, idx):
        labels = batch['label']
        out = self(batch)
        pred = torch.argmax(out.logits, dim=1)
        self.f1(pred, labels)
        self.acc(pred, labels)

        return {'val_loss': out.loss}

    def validation_epoch_end(self, outputs) -> None:
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log('val_acc', self.acc, logger=True)
        self.log('val_f1', self.f1, logger=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        """
            For the customed CheckpointIO
        """
        checkpoint['hg_model'] = self.model