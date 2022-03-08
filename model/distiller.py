from model.adpator import *
from argparse import ArgumentParser
from pytorch_lightning import LightningModule

import torch
import torchmetrics
import torch.nn.functional as F


class BaseDistiller(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Training Configurations
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--eps", default=1e-8, type=float)
        parser.add_argument("--num_classes", default=2, type=int)

        # Distillation Configurations
        parser.add_argument("--loss_list", default=['pred:mse'], type=list)

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

        loss_dict['pred:nll'] = out_s.loss
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
        # TO-DO not always have label
        self.teacher.eval()
        x, labels = batch['sentence'], batch['label']
        teacher_out = self.teacher(**x, labels=labels)
        student_out = self.student(**x, labels=labels)
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

        fn_lambda = lambda epoch: 0.85 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [fn_lambda, fn_lambda])
        return [optimizer], [scheduler]

    def training_step(self, batch, idx):
        loss_dict, nll_t = self.compute_loss(*self(batch))

        for k, v in loss_dict.items():
            self.log(k, v, on_step=True, on_epoch=False, prog_bar=True, logger=True)

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

