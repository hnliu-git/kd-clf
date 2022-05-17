
from argparse import ArgumentParser
from typing import Any, Dict, Optional
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities.types import _PATH
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.utilities.cloud_io import get_filesystem

import os
import torch
import torchmetrics


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
        checkpoint['student'].save_pretrained(path)

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


class BaseDistiller(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Training Configurations
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--epochs", default=5, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--eps", default=1e-8, type=float)
        parser.add_argument("--num_classes", default=2, type=int)

        # Distillation Configurations
        parser.add_argument("--temperature", default=4, type=float)
        parser.add_argument("--flood", default=0.07, type=float)

        return parser

    def __init__(self, teacher, student, args, adaptors):
        super().__init__()

        self.save_hyperparameters(args)

        self.teacher = teacher
        self.student = student
        self.adaptors = adaptors

        # Metrics
        self.acc_s = torchmetrics.Accuracy(num_classes=args.num_classes)
        self.f1_s = torchmetrics.F1Score(num_classes=args.num_classes)

    def compute_loss(self, out_t, out_s, mask=None):
        loss_dict = {
            'pred:nll': out_s.get('loss', 0),
            'nll_loss_teacher': out_t.get('loss', 0)
        }

        for adaptor in self.adaptors:
            feature_name = adaptor.name.split(':')[0]
            loss_dict[adaptor.name] = adaptor.w * adaptor(out_t.get(feature_name),
                                                          out_s.get(feature_name),
                                                          mask=mask)

        return loss_dict

    def forward(self, batch):
        """
        :param
            batch: {
                    sentence: dict from tokenizers
                    label: Tensor [bsz]
                   }
        :return:
            teacher_out: SequenceClassfierOutput with keys [logits]
            student_out: SequenceClassfierOutput with keys [loss, logits]
        """
        self.teacher.eval()
        with torch.no_grad():
            teacher_out = self.teacher(**batch)

        student_out = self.student(**batch)

        return teacher_out, student_out

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        parameters = [(n, p) for n, p in self.named_parameters() if 'teacher' not in n]

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
                                      eps=self.hparams.eps, )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=self.hparams.num_training_steps,
            num_warmup_steps=self.hparams.num_warmup_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, idx):
        loss_dict = self.compute_loss(
            *self(batch),
            batch.get('attention_mask')
        )

        if loss_dict['pred:nll'] == 0:
            loss_dict.pop('pred:nll')
            loss_dict.pop('nll_loss_teacher')
        elif self.global_step < int(self.hparams.num_training_steps / 3):
            # Temp pop prediction layer losses
            loss_dict.pop('pred:nll')
            if 'logit:ce' in loss_dict: loss_dict.pop('logit:ce')
            if 'logit:mse' in loss_dict: loss_dict.pop('logit:mse')
        else:
            # Flooding
            loss_dict['pred:nll'] = torch.abs(loss_dict['pred:nll'] - self.hparams.flood) + self.hparams.flood

        for k, v in loss_dict.items():
            self.log(k, v, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if 'nll_loss_teacher' in loss_dict:
            loss_dict.pop('nll_loss_teacher')

        return sum(loss_dict.values())

    def validation_step(self, batch, idx):
        labels = batch['labels']
        _, out_s = self(batch)
        pred_s = torch.argmax(out_s.logits, dim=1)

        self.f1_s(pred_s, labels)
        self.acc_s(pred_s, labels)

        return {'val_loss': out_s.loss}

    def test_step(self, batch, idx):
        labels = batch['labels']
        _, out_s = self(batch)
        pred_s = torch.argmax(out_s.logits, dim=1)

        self.f1_s(pred_s, labels)
        self.acc_s(pred_s, labels)

    def validation_epoch_end(self, outputs) -> None:
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log('val_f1', self.f1_s)
        self.log('val_acc', self.acc_s)

    def on_save_checkpoint(self, checkpoint) -> None:
        """
            For the customed CheckpointIO
        """
        checkpoint['student'] = self.student
