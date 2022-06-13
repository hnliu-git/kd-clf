import datasets
import yaml
import pytorch_lightning as pl

from helper.adaptor import *
from data.data_module import ClfDataModule
from helper.distiller import *

from utils import *
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets import Dataset
def get_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Teacher Model
    parser.add_argument("--teacher_model", default='bert-base-uncased', type=str,
                        help="name of the teacher model")

    # Student Model
    parser.add_argument("--student_model", default=None, type=str,
                        help="pretrained student model, the default is the bert_tiny model")
    parser.add_argument("--ckpt_path", default='ckpts', type=str)

    # Data configs
    parser.add_argument("--trn_dataset", default=None, type=str, required=True)
    parser.add_argument("--val_dataset", default=None, type=str, required=True)

    # Adaptor configs
    parser.add_argument('--adaptors', default=[], type=list)

    parser = ClfDataModule.add_model_specific_args(parser)
    parser = BaseDistiller.add_model_specific_args(parser)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pl.seed_everything(2022)
    args = get_args('configs/distillation.yaml')

    # Data Module
    dm = ClfDataModule(get_clf_dataset_obj(args), args)

    # Setup student and teacher
    teacher = path_to_clf_model(args.teacher_model, args.num_classes)
    student = path_to_clf_model(args.student_model, args.num_classes)

    str2adaptors = {
        'LogitMSE': LogitMSE(args.temperature),
        'LogitCE': LogitCE(args.temperature),
        'AttnTinyBERT': AttnTinyBERT(),
        'HidnTinyBERT': HidnTinyBERT(teacher.config.hidden_size, student.config.hidden_size),
        'EmbdTinyBERT': EmbdTinyBERT(teacher.config.hidden_size, student.config.hidden_size),
        'AttnMiniLM': AttnMiniLM(),
        'ValMiniLM': ValMiniLM(),
        'HidnPKD': HidnPKD(teacher.config.hidden_size, student.config.hidden_size),
    }


    # Setup adaptors
    adaptors = torch.nn.ModuleList([
        str2adaptors[name] for name in args.adaptors if name in str2adaptors
    ])

    # Setup lightning
    distiller = InterDistiller(
        teacher,
        student,
        args,
        adaptors,
    )

    logger = WandbLogger(project=args.project, name=args.exp)

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        monitor='attentions:mse',
        save_last=True,
        filename="1st-%s-%s"
                 % (args.val_dataset, args.student_model.split('/')[-1]),
    )

    trainer = Trainer(
        gpus=1,
        logger=logger,
        plugins=[HgCkptIO()],
        max_epochs=args.epochs,
        callbacks=[
            ckpt_callback,
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(distiller, dm)

    if args.val_dataset == 'tweet':
        trainer.test(distiller, dm)
    elif args.val_dataset == 'sst2':
        preds = trainer.predict(distiller, dataloaders=dm.test_dataloader())
        preds = [i for batch in preds for i in batch.numpy().tolist()]
        with open(f'data/{args.exp}-sst2.txt', 'w') as wf:
            for pred in preds:
                wf.write(f'{pred}\n')
