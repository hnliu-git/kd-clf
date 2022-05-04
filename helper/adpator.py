import math

import torch
from torch import nn

import torch.nn.functional as F


class LogitMSE(nn.Module):

    def __init__(self, temperature=1.0, name='logits:mse', w=1):
        """
        :param temperature: A float
        """
        super().__init__()
        self.temp = temperature
        self.name = name
        self.w = w

    def __call__(self, logits_t, logits_s, **kwargs):
        '''
        Calculate the mse loss between logits_s and logits_t

        :param logits_s: Tensor of shape (batch_size, num_labels)
        :param logits_t: Tensor of shape (batch_size, num_labels)

        '''
        logits_t = logits_t / self.temp
        logits_s = logits_s / self.temp
        loss = F.mse_loss(logits_s, logits_t)
        return loss


class LogitCE(nn.Module):

    def __init__(self, temperature=1.0, name='logits:ce', w=1):
        """
        :param temperature: A float
        """
        super().__init__()
        self.temp = temperature
        self.name = name
        self.w = w

    def __call__(self, logits_t, logits_s, **kwargs):
        '''
        Calculate the mse loss between logits_s and logits_t

        :param logits_s: Tensor of shape (batch_size, num_labels)
        :param logits_t: Tensor of shape (batch_size, num_labels)

        '''
        logits_t = logits_t / self.temp
        logits_s = logits_s / self.temp
        p_T = F.softmax(logits_t, dim=-1)
        loss = -(p_T * F.log_softmax(logits_s, dim=-1)).sum(dim=-1).mean()
        return loss


class AttnTinyBERT(nn.Module):

    def __init__(self, name='attentions:mse', w=1):
        super().__init__()
        self.w = w
        self.name = name

    def __call__(self, attn_t, attn_s, mask=None):
        '''
        * Calculates the clip mse loss between `tuple_t` and `tuple_s`.
        * Suitable when the two tuples have different lengths
        * The function will only calculate between the first len(tuple_s) tensors of tuple_t and tuple_s

        :param Tuple tuple_t: contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        :param Tuple tuple_s: contains tensor sof shape  (*batch_size*, *num_heads*, *length*, *length*)
        '''
        s_len = len(attn_s)
        attn_t = torch.cat(attn_t[-s_len:])
        attn_s = torch.cat(attn_s)

        if mask is None:
            attn_t = torch.where(attn_t <= -1e-3, torch.zeros_like(attn_t), attn_t)
            attn_s = torch.where(attn_s <= -1e-3, torch.zeros_like(attn_s), attn_s)
            loss = F.mse_loss(attn_s, attn_t)
        else:
            mask = torch.cat([mask for _ in range(s_len)])
            mask = mask.to(attn_s).unsqueeze(1).expand(-1, attn_s.size(1), -1)  # (bs, num_of_heads, len)
            valid_count = torch.pow(mask.sum(dim=2), 2).sum()
            loss = (F.mse_loss(attn_s, attn_t, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
                2)).sum() / valid_count

        return loss


class HidnTinyBERT(nn.Module):

    def __init__(self, hidn_sz_t, hidn_sz_s, name='hidden_states:mse', w=1):
        super().__init__()
        self.linear = torch.nn.Linear(hidn_sz_s, hidn_sz_t, bias=False)
        self.name = name
        self.w = w

    def __call__(self, hidn_t, hidn_s, mask=None):
        s_len = len(hidn_s)
        hidn_t = torch.cat((hidn_t[0],) + hidn_t[-s_len+1:])
        hidn_s = torch.cat(hidn_s)
        hidn_s = self.linear(hidn_s)

        if mask is None:
            loss = F.mse_loss(hidn_s, hidn_t)
        else:
            mask = torch.cat([mask for _ in range(s_len)])
            mask = mask.to(hidn_s)
            valid_count = mask.sum() * hidn_s.size(-1)
            loss = (F.mse_loss(hidn_s, hidn_t, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count

        return loss


class AttnMiniLM:

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


class HidnReln:

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




