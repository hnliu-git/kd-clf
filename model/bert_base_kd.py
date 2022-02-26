from argparse import ArgumentParser

import torch
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    AutoModelForSequenceClassification
)


class BertBaseKD(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--teacher_model", default='bert-base-uncased', type=str,
                            help="name of the teacher model")
        parser.add_argument("--hidden_size", default=768, type=int,
                            help="Dim of the encoder layer and pooler layer")
        parser.add_argument("--hidden_layers", default=12, type=int,
                            help="Number of hidden layers in encoder")
        parser.add_argument("--atten_heads", default=12, type=int,
                            help="Number of attention heads")
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--eps", default=1e-8, type=float)
        parser.add_argument("--num_classes", default=2, type=int)

        return parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # Setting up student
        config = BertConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.hidden_layers,
            num_attention_heads=args.atten_heads
        )
        self.student = BertForSequenceClassification(config)
        self.teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model)
        self.teacher.eval()

        # Metrics
        self.s_acc = torchmetrics.Accuracy(num_classes=args.num_classes)
        self.s_f1 = torchmetrics.F1Score(num_classes=args.num_classes)
        self.t_acc = torchmetrics.Accuracy(num_classes=args.num_classes)
        self.t_f1 = torchmetrics.F1Score(num_classes=args.num_classes)

    def _calculate_loss(self, t_out, s_out):
        s_nll_loss = s_out.loss
        t_nll_loss = t_out.loss

        t_score = t_out.logits
        s_score = s_out.logits
        mse_loss = F.mse_loss(t_score, s_score)
        return s_nll_loss, t_nll_loss, mse_loss

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
        x, labels = batch['sentence'], batch['label']
        teacher_out = self.teacher(**x, labels=labels)
        student_out = self.student(**x, labels=labels)
        return teacher_out, student_out

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=self.hparams.learning_rate,
                                      eps=self.hparams.eps, )

        fn_lambda = lambda epoch: 0.85 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [fn_lambda, fn_lambda])
        return [optimizer], [scheduler]

    def training_step(self, batch, idx):
        s_nll_loss, t_nll_loss, mse_loss = self._calculate_loss(*self(batch))
        self.log('s_nll_loss', s_nll_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('t_nll_loss', t_nll_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return t_nll_loss + mse_loss

    def validation_step(self, batch, idx):
        labels = batch['label']
        t_out, s_out = self(batch)
        s_pred = torch.argmax(s_out.logits, dim=1)
        t_pred = torch.argmax(t_out.logits, dim=1)
        self.s_f1(s_pred, labels)
        self.s_acc(s_pred, labels)
        self.t_f1(t_pred, labels)
        self.t_acc(t_pred, labels)

    def validation_epoch_end(self, outputs) -> None:
        self.log('acc_T_epoch', self.t_acc)
        self.log('f1_T_epoch', self.t_f1)
        self.log('acc_S_epoch', self.s_acc)
        self.log('f1_S_epoch', self.s_f1)
        self.teacher.eval()
