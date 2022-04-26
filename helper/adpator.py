import math

import torch
from torch import nn


class AttnMiniLMAdaptor:

    def __init__(self):
        pass

    def __call__(self, attn_t, attn_s):
        """
        MiniLM method: only calculate the attention loss of the last layer, use KL Divergence
        :param Tuple attn_t: contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        :param Tuple attn_s: contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        :return:
        """
        attn_t = torch.cat(attn_t[-1:])
        attn_s = torch.cat(attn_s[-1:])
        attn_t = torch.where(attn_t <= -1e-3, torch.zeros_like(attn_t), attn_t)
        attn_s = torch.where(attn_s <= -1e-3, torch.zeros_like(attn_s), attn_s)

        return attn_t, attn_s


class AttnAdaptor:

    def __init__(self):
        pass

    def __call__(self, attn_t, attn_s):
        '''
        * Calculates the clip mse loss between `tuple_t` and `tuple_s`.
        * Suitable when the two tuples have different lengths
        * The function will only calculate between the first len(tuple_s) tensors of tuple_t and tuple_s

        :param Tuple tuple_t: contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        :param Tuple tuple_s: contains tensor sof shape  (*batch_size*, *num_heads*, *length*, *length*)
        '''
        attn_t = torch.cat(attn_t[-len(attn_s):])
        attn_s = torch.cat(attn_s)
        attn_t = torch.where(attn_t <= -1e-3, torch.zeros_like(attn_t), attn_t)
        attn_s = torch.where(attn_s <= -1e-3, torch.zeros_like(attn_s), attn_s)

        return attn_t, attn_s


class HidnAdaptor(nn.Module):

    def __init__(self, config, hidden_size_t=768):
        super().__init__()
        self.w = torch.nn.Linear(config.hidden_size, hidden_size_t, bias=False)

    def __call__(self, hidn_t, hidn_s):
        hidn_t = torch.cat(hidn_t[-len(hidn_s):])
        hidn_s = torch.cat(hidn_s)
        hidn_s = self.w(hidn_s)

        return hidn_t, hidn_s


class HidnRelnAdaptor():

    def __init__(self):
        pass

    def __call__(self, hidn_t, hidn_s):
        hidn_t = hidn_t[-1]
        hidn_s = hidn_s[-1]

        batch, head, seq, _ = hidn_t.size()

        hidn_t = hidn_t.reshape(batch * head, seq, hidn_t.size()[-1])
        hidn_s = hidn_s.reshape(batch * head, seq, hidn_s.size()[-1])

        # reln_t = torch.einsum('b i d, b j d -> b i j', hidn_t, hidn_t) / hidn_t.size(2)
        # reln_s = torch.einsum('b i d, b j d -> b i j', hidn_s, hidn_s) / hidn_s.size(2)
        reln_t = torch.bmm(hidn_t, hidn_t.transpose(1, 2)) / math.sqrt(hidn_t.size()[-1])
        reln_s = torch.bmm(hidn_s, hidn_s.transpose(1, 2)) / math.sqrt(hidn_s.size()[-1])

        reln_t = torch.softmax(reln_t, dim=-1)
        reln_s = torch.softmax(reln_s, dim=-1)

        return reln_t, reln_s




