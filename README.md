# Transformer Distiller

## Introduction
The Transformer Distiller is a framework for knowledge distillation on Transformer-based models inspired by TextBrewer. 
Currently, it implements distillation methods presented in TinyBERT, MiniLM, and Patient-KD. 

Part of this work is in my master thesis in which I propose a two-step distilaltion method seperating prediction and 
intermediate layer distillation and prove it produce a robust and effective student model by experiments.
## Usage

### Adaptor Options
- `LogitMSE`
- `LogitCE`
- `AttnTinyBERT`
- `HidnTinyBERT`
- `EmbdTinyBERT`
- `AttnMiniLM`
- `ValMiniLM`
- `HidnPKD`
- ......

### Perform Knowledge Distillation

#### Setup configs in [distillation.yaml](configs/distillation.yaml)
```yaml
distillation:
  epochs: 10
  learning_rate: 3e-5
  adaptors:
    - '<adaptor1>' 
    - '<adaptor2>'
wandb:
  project: test
  exp: test
teacher_model:  ckpts/bert-base-uncased-epoch=02-val_loss=0.22
student_model: ckpts/bert_H192_L4_A12-epoch=19-perplexity=14.60
ckpt_path: ckpts/
```
#### Run the scripts

- Distillation on prediction and intermediate layer
```shell
python distillation_both.py
```

- Distillation on prediction layer
```shell
python distillation_pred.py
```

- Distillation on intermediate layer
```shell
python distillation_inter.py
```


## Comments

### Modification on Transformers Library

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
