# Transformer Distiller

## Introduction
The Transformer Distiller is a framework for knowledge distillation on Transformer-based models inspired by TextBrewer. 
Currently, it implements distillation methods presented in TinyBERT, MiniLM, and Patient-KD. 

Part of this work is in my master thesis in which I propose a two-step distilaltion method seperating prediction and 
intermediate layer distillation and prove it produces a robust and effective student model by experiments.

## Workflows

### Update the Transformer library
We modify the source code to output some intermeidate behaviors of Transformer.
A quick modification to the library can be achived by
```shell
python update_transformers.py
```


### Finetune
Task-specific distillation usually starts with fine-tuning a teacher model.
[`finetuner`](helper/finetuner.py) is provided to help with the first step. 
Below shows a short script on how to use it to fine-tune a BERT model on tweet dataset.

In [`finetune.py`](finetune.py), you will find how to configure finetune experiments using [finetune.yaml](configs/finetune.yaml). 

```python
from datasets import load_dataset
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from helper.finetuner import ClfFinetune, HgCkptIO
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSequenceClassification

model_name = 'bert-base-uncased'

dataset = load_dataset('tweet_eval', 'sentiment')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
dm = ClfDataModule(dataset, tokenizer=model_name)

fintuner = ClfFinetune(model, dm)

trainer = Trainer(
    plugins=[HgCkptIO()],
    callbacks=[ModelCheckpoint(monitor='val_loss',mode='min')]
)

trainer.fit(fintuner, dm)
```

### Mix-Step Distillation
The mix-step distillation refers to the original distillation method on Transformer models.
Both intermediate layers and the prediction layer will be updated by the defined adaptors.
[`distiller`](helper/distiller.py) is used to perform distillation, below shows a short script on how to use it to 
distill a fine-tuned BERT model into a compact student model.

In [`distillation_ms.py`](distillation.py), you will find how to configure distillation experiments using [distillation.yaml](configs/distillation.yaml).

```python
from datasets import load_dataset
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from helper.distiller import BaseDistiller, HgCkptIO
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSequenceClassification

def get_model(name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.output_values = True

    return model

teacher_model = '/path-to-teacher/'
student_model = 'huawei-noah/TinyBERT_General_4L_312D'

adaptors = [
  'AttnMiniLM',
  'ValMiniLM',
  'LogitMSE',
]

dataset = load_dataset('tweet_eval', 'sentiment')
teacher = get_model(teacher_model, num_labels=3)
student = get_model(student_model, num_labels=3)

dm = ClfDataModule(dataset, tokenizer=teacher_model)

distiller = BaseDistiller(teacher, student, adaptors, dm)

trainer = Trainer(
    plugins=[HgCkptIO()],
    callbacks=[ModelCheckpoint(monitor='val_loss',mode='min')]
)

trainer.fit(distiller, dm)
```

### Two-Step distillation
Two-step distillation is a method we proposed in the project to alleviate over-fitting issue.
The method split distillaiton into two steps, the first step only update the intermediate layers while the second step
only update the prediction layer. Below gives an example on how to perform the two-step method.

To use `wandb` to monitor the experiments, we have to separate the two steps into two files [distillation_ts_1st.py](distillation_ts_1st.py)
and [distillation_ts_2nd.py](distillation_ts_2nd.py) or it will lead to unexpected behaviors.

```python
from datasets import load_dataset
from pytorch_lightning import Trainer
from data.data_module import ClfDataModule
from helper.distiller import InterDistiller, PredDistiller, HgCkptIO
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSequenceClassification

def get_model(name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.output_values = True

    return model

teacher_model = 'huawei-noah/TinyBERT_General_4L_312D'
student_model = 'huawei-noah/TinyBERT_General_4L_312D'

adaptors_1 = ['AttnMiniLM','ValMiniLM',]
adaptors_2 = ['LogitMSE']

dataset = load_dataset('tweet_eval', 'sentiment')
teacher = get_model(teacher_model, num_labels=3)
student = get_model(student_model, num_labels=3)

dm = ClfDataModule(dataset, tokenizer=teacher_model)

distiller_1 = InterDistiller(teacher, student, adaptors_1, dm)

trainer_1 = Trainer(
    plugins=[HgCkptIO()],
    callbacks=[ModelCheckpoint(dirpath='student_1st', monitor='val_loss',mode='min', save_last=True)]
)

trainer_1.fit(distiller_1, dm)

distiller_2 = PredDistiller(teacher, student, ['LogitMSE'], dm)

trainer_2 = Trainer(
    plugins=[HgCkptIO()],
    callbacks=[ModelCheckpoint(dirpath='student_2nd', monitor='val_loss',mode='min', save_last=True)]
)

trainer_2.fit(distiller_2, dm)
```


## Comments

### Modification on Transformers Library
To produce intermediate behaviours from Transformer models, we modified the source code of Transformers library.

Below highlights the changes to the source code
- [configuration_utils.py](bert/configuration_utils.py)
  - Line 257
- [configuration_bert.py](bert/configuration_bert.py)
  - `BertConfig`
- [modeling_outputs.py](bert/modeling_outputs.py)
  - `SequenceClassifierOutput`
  - `BaseModelOutputWithPoolingAndCrossAttentions`
  - `BaseModelOutputWithPastAndCrossAttentions`
- [modeling_bert.py](bert/modeling_bert.py)
    - Add `output_values` parameter
    - Output attention matrices before dropout in `BertSelfAttention`
    - Modified `BertModel` so that it can collect `values`
    - Modified `BertSelfAttention` so that it output `value`

## Citations
TextBrewer
```bibtex
@InProceedings{textbrewer-acl2020-demo,
    title = "{T}ext{B}rewer: {A}n {O}pen-{S}ource {K}nowledge {D}istillation {T}oolkit for {N}atural {L}anguage {P}rocessing",
    author = "Yang, Ziqing and Cui, Yiming and Chen, Zhipeng and Che, Wanxiang and Liu, Ting and Wang, Shijin and Hu, Guoping",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.2",
    pages = "9--16",
}
```
TinyBERT
```bibtex
@article{jiao2019tinybert,
  title={Tinybert: Distilling bert for natural language understanding},
  author={Jiao, Xiaoqi and Yin, Yichun and Shang, Lifeng and Jiang, Xin and Chen, Xiao and Li, Linlin and Wang, Fang and Liu, Qun},
  journal={arXiv preprint arXiv:1909.10351},
  year={2019}
}
```
MiniLM
```bibtex
@article{wang2020minilm,
  title={Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers},
  author={Wang, Wenhui and Wei, Furu and Dong, Li and Bao, Hangbo and Yang, Nan and Zhou, Ming},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={5776--5788},
  year={2020}
}
```
Patient Knowledge Distillation
```bibtex
@article{sun2019patient,
  title={Patient knowledge distillation for bert model compression},
  author={Sun, Siqi and Cheng, Yu and Gan, Zhe and Liu, Jingjing},
  journal={arXiv preprint arXiv:1908.09355},
  year={2019}
}
```
