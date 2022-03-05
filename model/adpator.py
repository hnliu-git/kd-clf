
import torch
from torch.nn import Module


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
        attn_t = torch.cat(attn_t[:len(attn_s)])
        attn_s = torch.cat(attn_s)
        # attn_t = torch.where(attn_t <= -1e-3, torch.zeros_like(attn_t), attn_t)
        # attn_s = torch.where(attn_s <= -1e-3, torch.zeros_like(attn_s), attn_s)

        return attn_t, attn_s


class HidnAdaptor(Module):

    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.ParameterDict({
            "hidn": torch.nn.Parameter(w)
        })

    def __call__(self, hidn_t, hidn_s):
        hidn_t = torch.cat(hidn_t[:len(hidn_s)])
        hidn_s = torch.cat(hidn_s)
        hidn_t = torch.matmul(hidn_t, self.w['hidn'])

        return hidn_t, hidn_s




