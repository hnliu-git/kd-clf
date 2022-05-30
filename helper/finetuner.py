from argparse import ArgumentParser
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import _PATH

import os
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import CheckpointIO
from transformers import get_linear_schedule_with_warmup

from pytorch_lightning.utilities.cloud_io import get_filesystem


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
        checkpoint['hg_model'].save_pretrained(path)

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

        parser.add_argument('--epochs', default=5, type=int)
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--eps", default=1e-8, type=float)
        parser.add_argument("--num_classes", default=2, type=int)

        return parser

    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model

        # Metrics
        self.acc = torchmetrics.Accuracy(num_classes=self.hparams.num_classes)
        self.f1 = torchmetrics.F1Score(num_classes=self.hparams.num_classes)

    def forward(self, batch):
        return self.model(**batch)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        parameters = [(n, p) for n, p in self.named_parameters()]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=self.hparams.learning_rate,
                                      eps=self.hparams.eps,)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=self.hparams.num_training_steps,
            num_warmup_steps=self.hparams.num_warmup_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, idx):
        loss = self(batch).loss
        self.log("pred:nll", loss, logger=True)
        return torch.abs(loss-0.1) + 0.1

    def validation_step(self, batch, idx):
        labels = batch['labels']
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
            For the customized CheckpointIO
        """
        checkpoint['hg_model'] = self.model