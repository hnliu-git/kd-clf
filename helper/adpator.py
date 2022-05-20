import math

import torch
from torch import nn

import torch.nn.functional as F
import wandb


class LogitMSE(nn.Module):

    def __init__(self, temperature=4.0, name='logits:mse', w=1):
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

    def __init__(self, temperature=4.0, name='logits:ce', w=1):
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
        attn_t = torch.where(attn_t <= -1e-3, torch.zeros_like(attn_t), attn_t)
        attn_s = torch.where(attn_s <= -1e-3, torch.zeros_like(attn_s), attn_s)

        if mask is None:
            loss = F.mse_loss(attn_s, attn_t)
        else:
            mask = torch.cat([mask for _ in range(s_len)])
            mask = mask.to(attn_s).unsqueeze(1).expand(-1, attn_s.size(1), -1)  # (bs, num_of_heads, len)
            valid_count = torch.pow(mask.sum(dim=2), 2).sum()

            attn_t_masked = (attn_t * mask.unsqueeze(-1) * mask.unsqueeze(2))
            attn_s_masked = (attn_s * mask.unsqueeze(-1) * mask.unsqueeze(2))

            loss = F.mse_loss(attn_s_masked, attn_t_masked, reduction='none').sum() / valid_count

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


class EmbdTinyBERT(nn.Module):

    def __init__(self, hidn_sz_t, hidn_sz_s, name='hidden_states:embd:mse', w=1):
        super().__init__()
        self.linear = torch.nn.Linear(hidn_sz_s, hidn_sz_t, bias=False)
        self.name = name
        self.w = w

    def __call__(self, hidn_t, hidn_s, mask=None):
        hidn_t = hidn_t[0]
        hidn_s = hidn_s[0]
        hidn_s = self.linear(hidn_s)

        if mask is None:
            loss = F.mse_loss(hidn_s, hidn_t)
        else:
            mask = mask.to(hidn_s)
            valid_count = mask.sum() * hidn_s.size(-1)
            loss = (F.mse_loss(hidn_s, hidn_t, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count

        return loss


class AttnMiniLM(nn.Module):

    def __init__(self, name='attentions:ce', w=1):
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
        attn_t = torch.cat(attn_t[-1:])
        attn_s = torch.cat(attn_s[-1:])

        if mask is None:
            attn_t = torch.where(attn_t <= -1e-3, torch.zeros_like(attn_t), attn_t)
            loss = -((attn_t * F.log_softmax(attn_s, dim=-1)).sum(dim=-1)).mean()
        else:
            mask = mask.to(attn_s).unsqueeze(1).expand(-1, attn_s.size(1), -1)  # (bs, num_of_heads, len)
            # Smooth as the masked feature is zero here
            attn_s = (attn_s + 1e-6) / (attn_s.sum() + attn_s.size(-1) * 1e-6)
            attn_t = (attn_t + 1e-6) / (attn_t.sum() + attn_t.size(-1) * 1e-6)
            # Attn_t and Attn_s are already applied softmax
            loss = (-(attn_t * attn_s.log()) * mask.unsqueeze(-1) * mask.unsqueeze(2)).sum() / mask.sum()

        return loss


class ValMiniLM(nn.Module):

    def __init__(self, name='values:ce', w=1):
        super().__init__()
        self.name = name
        self.w = w

    def __call__(self, val_t, val_s, mask=None):
        val_t = torch.cat(val_t[-1:])
        val_s = torch.cat(val_s[-1:])

        batch, head, seq, t_dim = val_t.size()
        _, _, _, s_dim = val_s.size()

        if mask is None:
            val_t = val_t.reshape(batch * head, seq, t_dim)
            val_s = val_s.reshape(batch * head, seq, s_dim)

            reln_t = torch.bmm(val_t, val_t.transpose(1, 2)) / math.sqrt(t_dim)
            reln_s = torch.bmm(val_s, val_s.transpose(1, 2)) / math.sqrt(s_dim)

            prob_t = F.softmax(reln_t, dim=-1)
            loss = -((prob_t * F.log_softmax(reln_s, dim=-1)).sum(dim=-1)).mean()
        else:
            mask = torch.cat([mask for _ in range(head)]).to(val_t)

            val_t = val_t.reshape(batch * head, seq, t_dim)
            val_s = val_s.reshape(batch * head, seq, s_dim)

            reln_t = torch.bmm(val_t, val_t.transpose(1, 2)) / math.sqrt(t_dim)
            reln_s = torch.bmm(val_s, val_s.transpose(1, 2)) / math.sqrt(s_dim)

            prob_t = F.softmax(reln_t, dim=-1)
            loss = -((prob_t * F.log_softmax(reln_s, dim=-1) * mask.unsqueeze(1)).sum(dim=-1)*mask).sum() / mask.sum()

        return loss


class AttnMiniLMMSE(nn.Module):

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
        attn_t = torch.cat(attn_t[-1:])
        attn_s = torch.cat(attn_s[-1:])

        if mask is None:
            attn_t = torch.where(attn_t <= -1e-3, torch.zeros_like(attn_t), attn_t)
            attn_s = torch.where(attn_s <= -1e-3, torch.zeros_like(attn_s), attn_s)
            loss = F.mse_loss(attn_s, attn_t)
        else:
            mask = mask.to(attn_s).unsqueeze(1).expand(-1, attn_s.size(1), -1)  # (bs, num_of_heads, len)
            valid_count = torch.pow(mask.sum(dim=2), 2).sum()

            attn_t_masked = (attn_t * mask.unsqueeze(-1) * mask.unsqueeze(2))
            attn_s_masked = (attn_s * mask.unsqueeze(-1) * mask.unsqueeze(2))

            loss = F.mse_loss(attn_s_masked, attn_t_masked, reduction='none').sum() / valid_count

        return loss


class ValMiniLMMSE(nn.Module):

    def __init__(self, name='values:mse', w=1):
        super().__init__()
        self.name = name
        self.w = w

    def __call__(self, val_t, val_s, mask=None):
        val_t = torch.cat(val_t[-1:])
        val_s = torch.cat(val_s[-1:])

        batch, head, seq, t_dim = val_t.size()
        _, _, _, s_dim = val_s.size()

        if mask is None:
            val_t = val_t.reshape(batch * head, seq, t_dim)
            val_s = val_s.reshape(batch * head, seq, s_dim)

            reln_t = torch.bmm(val_t, val_t.transpose(1, 2)) / math.sqrt(t_dim)
            reln_s = torch.bmm(val_s, val_s.transpose(1, 2)) / math.sqrt(s_dim)

            loss = F.mse_loss(reln_s, reln_t)
        else:
            mask = torch.cat([mask for _ in range(head)]).to(val_t)

            val_t = val_t.reshape(batch * head, seq, t_dim) * mask.unsqueeze(-1)
            val_s = val_s.reshape(batch * head, seq, s_dim) * mask.unsqueeze(-1)

            reln_t = torch.bmm(val_t, val_t.transpose(1, 2)) / math.sqrt(t_dim)
            reln_s = torch.bmm(val_s, val_s.transpose(1, 2)) / math.sqrt(s_dim)

            valid_count = torch.pow(mask.sum(dim=1), 2).sum()
            loss = F.mse_loss(reln_s, reln_t, reduction='none').sum() / valid_count

        return loss


class HidnPKD(nn.Module):

    def __init__(self, hidn_sz_t, hidn_sz_s, name='hidden_states:mse', w=1):
        super().__init__()
        self.linear = torch.nn.Linear(hidn_sz_s, hidn_sz_t, bias=False)
        self.name = name
        self.w = w

    def __call__(self, hidn_t, hidn_s, mask=None):

        s_len = len(hidn_s)
        cls_t = torch.cat(hidn_t[-s_len:])[:, 0]
        cls_s = torch.cat(hidn_s)[:, 0]

        cls_s = self.linear(cls_s)

        cls_t = cls_t / torch.norm(cls_t, dim=1, keepdim=True)
        cls_s = cls_s / torch.norm(cls_s, dim=1, keepdim=True)

        loss = (cls_s - cls_t).pow(2).sum(dim=-1).mean()

        return loss





