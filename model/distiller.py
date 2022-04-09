
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
import torch.nn.functional as F


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
        checkpoint['student'].save_pretrained(path.replace('.ckpt', ''))

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
        parser.add_argument("--loss_list", default=None, type=list)
        return parser

    def __init__(self, teacher, student, args, attn_adaptor=None, hidn_adaptor=None):
        super().__init__()

        self.save_hyperparameters(args)

        self.teacher = teacher
        self.student = student
        self.attn_adaptor = attn_adaptor
        self.hidn_adaptor = hidn_adaptor

        # Metrics
        self.acc_s = torchmetrics.Accuracy(num_classes=args.num_classes)
        self.acc_t = torchmetrics.Accuracy(num_classes=args.num_classes)

        # loss functions
        self.loss_func = {
            'mse': F.mse_loss,
            'kld': F.kl_div
        }

    def compute_loss(self, out_t, out_s):
        loss_dict = {}
        for loss_name in self.hparams.loss_list:
            name, func = loss_name.split(':')
            if name == 'pred':
                if func == 'nll':
                    if out_s.loss:
                        loss_dict['pred:nll'] = out_s.loss
                    continue
                score_t = out_t.logits
                score_s = out_s.logits
                loss = self.loss_func[func](score_t, score_s)
                loss_dict[name+':'+func] = loss
            elif name == 'attn':
                attn_t = out_t.attentions
                attn_s = out_s.attentions
                tsr_t, tsr_s = self.attn_adaptor(attn_t, attn_s)
                loss = self.loss_func[func](tsr_t, tsr_s)
                loss_dict[name + ':' + func] = loss
            elif name == 'hidn':
                hidn_t = out_t.hidden_states[1:]
                hidn_s = out_s.hidden_states[1:]
                tsr_t, tsr_s = self.hidn_adaptor(hidn_t, hidn_s)
                loss = self.loss_func[func](tsr_t, tsr_s)
                loss_dict[name + ':' + func] = loss
            elif name == 'embd':
                embd_t = out_t.hidden_states[0:1]
                embd_s = out_s.hidden_states[0:1]
                tsr_t, tsr_s = self.hidn_adaptor(embd_t, embd_s)
                loss = self.loss_func[func](tsr_t, tsr_s)
                loss_dict[name + ':' + func] = loss

        nll_t = out_t.loss

        return loss_dict, nll_t

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

        if 'label' in batch:
            x, labels = batch['sentence'], batch['label']
            teacher_out = self.teacher(**x, labels=labels)
            student_out = self.student(**x, labels=labels)
        else:
            x = batch['sentence']
            teacher_out = self.teacher(**x)
            student_out = self.student(**x)

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
        loss_dict, nll_t = self.compute_loss(*self(batch))

        for k, v in loss_dict.items():
            self.log(k, v, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if nll_t:
            self.log('nll_loss_teacher', nll_t, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return sum(loss_dict.values())

    def validation_step(self, batch, idx):
        labels = batch['label']
        out_t, out_s = self(batch)
        pred_t = torch.argmax(out_t.logits, dim=1)
        pred_s = torch.argmax(out_s.logits, dim=1)

        self.acc_t(pred_t, labels)
        self.acc_s(pred_s, labels)

        return {'val_nll_loss': out_s.loss}

    def validation_epoch_end(self, outputs) -> None:
        val_loss = torch.stack([x["val_nll_loss"] for x in outputs]).mean()
        self.log("val_nll_loss", val_loss, prog_bar=True, logger=True)
        self.log('val_acc_t', self.acc_t)
        self.log('val_acc_s', self.acc_s)

    def on_save_checkpoint(self, checkpoint) -> None:
        """
            For the customed CheckpointIO
        """
        checkpoint['student'] = self.student