
## Introduction
This file gives an introduction of the goal of modification on `transformers` library and highlight the changes to the source code

## Goal
The native `transformers` library doesn't output the value matrices in the Attention mechanism. 
The main goal of the modfication is to output that behaviors.

## Highlights
- [configuration_utils.py](configuration_utils.py)
  - Line 257
- [configuration_bert.py](configuration_bert.py)
  - `BertConfig`
- [modeling_outputs.py](modeling_outputs.py)
  - `SequenceClassifierOutput`
  - `BaseModelOutputWithPoolingAndCrossAttentions`
  - `BaseModelOutputWithPastAndCrossAttentions`
- [modeling_bert.py](modeling_bert.py)
    - Add `output_values` parameter
    - Output attention matrices before dropout in `BertSelfAttention`
    - Modified `BertModel` so that it can collect `values`
    - Modified `BertSelfAttention` so that it output `value`
- [modeling_roberta.py](modeling_roberta.py)
  - Similar to the `modeling_bert.py`