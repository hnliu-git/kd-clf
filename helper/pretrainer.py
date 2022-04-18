from argparse import ArgumentParser
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import _PATH

import os
import math
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities.cloud_io import get_filesystem
from transformers import (
    BertConfig,
    BertForMaskedLM
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


class Pretrainer(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--ckpt_path", default='ckpts', type=str)

        # Model Configs
        parser.add_argument("--hidden_size", default=768, type=int,
                            help="Dim of the encoder layer and pooler layer of the student")
        parser.add_argument("--hidden_layers", default=12, type=int,
                            help="Number of hidden layers in encoder of the student")
        parser.add_argument("--atten_heads", default=12, type=int,
                            help="Number of attention heads of the student")

        # Training Configs
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--eps", default=1e-8, type=float)
        parser.add_argument("--num_classes", default=2, type=int)

        return parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        config = BertConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.hidden_layers,
            num_attention_heads=args.atten_heads
        )

        self.model = BertForMaskedLM(config)

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
        return self.model(**batch)

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

        def fn_lambda(epoch):
            if epoch < 1:
                return 0.01
            else:
                return 0.85 ** (epoch - 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[fn_lambda, fn_lambda])
        return [optimizer], [scheduler]

    def training_step(self, batch, idx):
        loss = self(batch).loss
        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, idx):
        return {'val_loss': self(batch).loss}

    def validation_epoch_end(self, outputs) -> None:
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        try:
            perplexity = math.exp(val_loss)
        except OverflowError:
            perplexity = float("inf")
        self.log("perplexity", perplexity, prog_bar=True, logger=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        """
            For the customed CheckpointIO
        """
        checkpoint['hg_model'] = self.model