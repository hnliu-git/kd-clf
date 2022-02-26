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
        parser.add_argument("--student_model", default=None, type=str,
                            help="pretrained student model, the default is the bert_tiny model")
        parser.add_argument("--hidden_size", default=768, type=int,
                            help="Dim of the encoder layer and pooler layer of the student")
        parser.add_argument("--hidden_layers", default=12, type=int,
                            help="Number of hidden layers in encoder of the student")
        parser.add_argument("--atten_heads", default=12, type=int,
                            help="Number of attention heads of the student")
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--eps", default=1e-8, type=float)
        parser.add_argument("--num_classes", default=2, type=int)

        return parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # Setting up student
        if args.student_model:
            self.student = AutoModelForSequenceClassification.from_pretrained(args.student_model)
        else:
            config = BertConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.hidden_layers,
                num_attention_heads=args.atten_heads
            )
            self.student = BertForSequenceClassification(config)

        self.teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model)

        # Metrics
        self.acc_s = torchmetrics.Accuracy(num_classes=args.num_classes)
        # self.s_f1 = torchmetrics.F1Score(num_classes=args.num_classes)
        self.acc_t = torchmetrics.Accuracy(num_classes=args.num_classes)
        # self.t_f1 = torchmetrics.F1Score(num_classes=args.num_classes)

    def _calculate_loss(self, out_t, out_s):
        nll_loss_s = out_s.loss
        nll_loss_t = out_t.loss

        score_t = out_t.logits
        score_s = out_s.logits
        mse_loss = F.mse_loss(score_t, score_s)
        return nll_loss_t, nll_loss_s, mse_loss

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

    def on_train_start(self) -> None:
        self.teacher.eval()

    def training_step(self, batch, idx):
        nll_loss_t, nll_loss_s, mse_loss = self._calculate_loss(*self(batch))
        self.log('nll_loss_t', nll_loss_t, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('nll_loss_s', nll_loss_s, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return nll_loss_s + mse_loss

    def validation_step(self, batch, idx):
        labels = batch['label']
        out_t, out_s = self(batch)
        pred_t = torch.argmax(out_t.logits, dim=1)
        pred_s = torch.argmax(out_s.logits, dim=1)
        # self.s_f1(s_pred, labels)
        # self.t_f1(t_pred, labels)
        self.acc_t(pred_t, labels)
        self.acc_s(pred_s, labels)

        return {'val_nll_loss': out_s.loss}

    def validation_epoch_end(self, outputs) -> None:
        val_loss = torch.stack([x["val_nll_loss"] for x in outputs]).mean()
        self.log("val_nll_loss", val_loss, prog_bar=True, logger=True)
        self.log('val_acc_t', self.acc_t)
        self.log('val_acc_s', self.acc_s)
        # self.log('f1_T_epoch', self.t_f1)
        # self.log('f1_S_epoch', self.s_f1)
