"""
Contains Adaptor Classes for teacher and student features adaption
"""

import math
import torch
import torch.nn.functional as F

from torch import nn


class BaseAdaptor(nn.Module):

    def __init__(self, name, w):
       """
       :param name: name of the adaptor
       :param w: weight for the result loss
       """
       super(BaseAdaptor, self).__init__()
       self.name = name
       self.w = w


class LogitMSE(BaseAdaptor):

    def __init__(self, temperature=4.0, name='logits:mse', w=1):
        """
        :param temperature: a float to soften the logits
        """
        super().__init__(name, w)
        self.temp = temperature

    def __call__(self, logits_t, logits_s, mask=None):
        '''
        Calculate the mse loss between logits_s and logits_t

        :param logits_s: Tensor of shape (batch_size, num_labels)
        :param logits_t: Tensor of shape (batch_size, num_labels)

        '''
        logits_t = logits_t / self.temp
        logits_s = logits_s / self.temp
        loss = F.mse_loss(logits_s, logits_t)
        return loss


class LogitCE(BaseAdaptor):

    def __init__(self, temperature=4.0, name='logits:ce', w=1):
        """
        :param temperature: a float to soften the logits
        """
        super().__init__(name, w)
        self.temp = temperature

    def __call__(self, logits_t, logits_s, mask=None):
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


class AttnTinyBERT(BaseAdaptor):

    def __init__(self, name='attentions:mse', w=1):
        super().__init__(name, w)

    def __call__(self, attn_t, attn_s, mask=None):
        '''
        * Calculates mse loss between `attn_t` and `attn_s` defined in TinyBERT.
        * The function will use a 'last' strategy which mean `attn_s`
          only learn from the last `len(attn_s)` layers from `attn_t`

        :param attn_t (Tuple): contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        :param attn_s (Tuple): contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        '''
        bsz, head, seq, seq = attn_t[0].size()
        s_len = len(attn_s)
        # The `last` strategy
        attn_t = torch.cat(attn_t[-s_len:])
        attn_s = torch.cat(attn_s)

        if mask is None:
            loss = F.mse_loss(attn_s, attn_t)
        else:
            norm_term = mask.sum() * s_len * head
            mask = torch.cat([mask for _ in range(s_len)])
            mask = mask.to(attn_s).unsqueeze(1).expand(-1, head, -1)  # (bs, num_of_heads, len)

            attn_t_masked = (attn_t * mask.unsqueeze(-1) * mask.unsqueeze(2))
            attn_s_masked = (attn_s * mask.unsqueeze(-1) * mask.unsqueeze(2))

            loss = F.mse_loss(attn_s_masked, attn_t_masked, reduction='none').sum() / norm_term

        return loss


class HidnTinyBERT(BaseAdaptor):

    def __init__(self, hidn_sz_t, hidn_sz_s, name='hidden_states:mse', w=1):
        super().__init__(name, w)
        self.linear = torch.nn.Linear(hidn_sz_s, hidn_sz_t, bias=False)

    def __call__(self, hidn_t, hidn_s, mask=None):
        '''
        * Calculates mse loss between `hidn_t` and `hidn_s` defined in TinyBERT.
        * The function will use a 'last' strategy which mean `hidn_s`
          only learn from the last `len(hidn_s)` layers from `hidn_t`

        :param hidn_t (Tuple): contains tensors of shape  (*batch_size*, *length*, *dim*)
        :param hidn_s (Tuple): contains tensors of shape  (*batch_size*, *length*, *dim*)
        '''
        bsz, seq, s_dim = hidn_s[0].size()
        s_len = len(hidn_s)

        hidn_t = torch.cat(hidn_t[-s_len:])
        hidn_s = torch.cat(hidn_s)
        hidn_s = self.linear(hidn_s)

        if mask is None:
            loss = F.mse_loss(hidn_s, hidn_t)
        else:
            norm_term = mask.sum() * s_len * s_dim
            mask = torch.cat([mask for _ in range(s_len)])
            mask = mask.to(hidn_s)
            loss = (F.mse_loss(hidn_s, hidn_t, reduction='none') * mask.unsqueeze(-1)).sum() / norm_term

        return loss


class EmbdTinyBERT(BaseAdaptor):

    def __init__(self, hidn_sz_t, hidn_sz_s, name='hidden_states:embd:mse', w=1):
        super().__init__(name, w)
        self.linear = torch.nn.Linear(hidn_sz_s, hidn_sz_t, bias=False)

    def __call__(self, hidn_t, hidn_s, mask=None):
        '''
        * Calculates mse loss between `hidn_t` and `hidn_s` defined in TinyBERT.
        * The function will only use the first hidden_state from student and teacher,
          aka embedding vectors

        :param hidn_t (Tuple): contains tensors of shape  (*batch_size*, *length*, *dim*)
        :param hidn_s (Tuple): contains tensors of shape  (*batch_size*, *length*, *dim*)
        '''
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


class AttnMiniLM(BaseAdaptor):

    def __init__(self, name='attentions:kld', w=1):
        super().__init__(name, w)

    def __call__(self, attn_t, attn_s, mask=None):
        '''
        * Calculates kld loss between `attn_t` and `attn_s` defined in MiniLM.
        * The function will use a 'last' strategy which mean `attn_s`
          only learn from the last `len(attn_s)` layers from `attn_t`

        :param attn_t (Tuple): contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        :param attn_s (Tuple): contains tensors of shape  (*batch_size*, *num_heads*, *length*, *length*)
        '''

        bsz, head, seq, seq = attn_s[0].size()
        s_len = len(attn_s)

        attn_t = torch.cat(attn_t[-s_len:])
        attn_s = torch.cat(attn_s)

        if mask is None:
            attn_s = attn_s + 1e-6 / attn_s.sum(dim=-1, keepdim=True)
            attn_t = attn_t + 1e-6 / attn_t.sum(dim=-1, keepdim=True)
            loss = F.kl_div(attn_s.log(), attn_t)
        else:
            norm_term = mask.sum() * s_len * head
            mask = torch.cat([mask for _ in range(s_len)])
            mask = mask.to(attn_s).unsqueeze(1).expand(-1, head, -1)

            # Smooth as some attention scores might be zero
            attn_s = attn_s + (mask * 1e-6).unsqueeze(2)
            attn_t = attn_t + (mask * 1e-6).unsqueeze(2)
            # Re-distribute and make sure the sum is 1
            attn_s = attn_s / attn_s.sum(dim=-1, keepdim=True)
            attn_t = attn_t / attn_t.sum(dim=-1, keepdim=True)
            # Avoid log(0) coz the mask positions have value 0
            attn_t += ((1 - mask) * 1e-6).unsqueeze(2)
            attn_s += ((1 - mask) * 1e-6).unsqueeze(2)

            kld_loss = F.kl_div(attn_s.log(), attn_t, reduction='none')
            loss = (kld_loss * mask.unsqueeze(-1) * mask.unsqueeze(2)).sum() / norm_term

        return loss


class ValMiniLM(BaseAdaptor):

    def __init__(self, name='values:kld', w=1):
        super().__init__(name, w)

    def __call__(self, val_t, val_s, mask=None):
        '''
           * Calculates kld loss between `val_t` and `val_s` defined in MiniLM.
           * The function will use a 'last' strategy which means `val_s`
             only learn from the last `len(val_s)` layers from `val_t`

           :param val_t (Tuple): contains tensors of shape  (*batch_size*, *num_heads*, *length*, *dim*)
           :param val_s (Tuple): contains tensors of shape  (*batch_size*, *num_heads*, *length*, *dim*)
        '''
        batch, head, seq, t_dim = val_t[0].size()
        _, _, _, s_dim = val_s[0].size()
        s_len = len(val_s)

        val_t = torch.cat(val_t[-s_len:])
        val_s = torch.cat(val_s)

        if mask is None:
            val_t = val_t.reshape(batch * head, seq, t_dim)
            val_s = val_s.reshape(batch * head, seq, s_dim)

            reln_t = torch.bmm(val_t, val_t.transpose(1, 2)) / math.sqrt(t_dim)
            reln_s = torch.bmm(val_s, val_s.transpose(1, 2)) / math.sqrt(s_dim)

            reln_t = F.softmax(reln_t, dim=-1)
            reln_s = F.softmax(reln_s, dim=-1)

            loss = F.kl_div(reln_s.log(), reln_t)
        else:
            norm_term = mask.sum() * s_len * head
            mask = torch.cat([mask for _ in range(s_len)])
            mask = torch.cat([mask for _ in range(head)]).to(val_t)
            # For masked items, Softmax(-inf) = 0
            mask_reversed = (1.0 - mask) * -10000.0

            val_t = val_t.reshape(batch * head * s_len, seq, t_dim)
            val_s = val_s.reshape(batch * head * s_len, seq, s_dim)

            reln_t = (torch.bmm(val_t, val_t.transpose(1, 2)) / math.sqrt(t_dim)) + mask_reversed.unsqueeze(1)
            reln_s = (torch.bmm(val_s, val_s.transpose(1, 2)) / math.sqrt(s_dim)) + mask_reversed.unsqueeze(1)

            reln_t = F.softmax(reln_t, dim=-1)
            reln_s = F.softmax(reln_s, dim=-1)

            # Smooth as the masked feature is zero here
            reln_s = reln_s + (mask * 1e-6).unsqueeze(2)
            reln_t = reln_t + (mask * 1e-6).unsqueeze(2)
            # Re-distribute and make sure the sum is 1
            reln_s = reln_s / reln_s.sum(dim=-1, keepdim=True)
            reln_t = reln_t / reln_t.sum(dim=-1, keepdim=True)
            # Avoid log(0) coz the mask positions have value 0
            reln_t += ((1 - mask) * 1e-6).unsqueeze(2)
            reln_s += ((1 - mask) * 1e-6).unsqueeze(2)

            kld_loss = F.kl_div(reln_s.log(), reln_t, reduction='none')
            loss = (kld_loss * mask.unsqueeze(1) * mask.unsqueeze(2)).sum() / norm_term

        return loss


class HidnPKD(BaseAdaptor):

    def __init__(self, hidn_sz_t, hidn_sz_s, name='hidden_states:mse', w=1):
        super().__init__(name, w)
        self.linear = torch.nn.Linear(hidn_sz_s, hidn_sz_t, bias=False)
        self.name = name
        self.w = w

    def __call__(self, hidn_t, hidn_s, mask=None):
        '''
            * Calculates mse loss between `hidn_t` and `hidn_s` defined in Patient-KD.
            * The function will use a 'last' strategy which mean `hidn_s`
              only learn from the last `len(hidn_s)` layers from `hidn_t`.
            * Only features of the first token [CLS] is used.

            :param hidn_t (Tuple): contains tensors of shape  (*batch_size*, *length*, *dim*)
            :param hidn_s (Tuple): contains tensors of shape  (*batch_size*, *length*, *dim*)
        '''
        s_len = len(hidn_s)
        cls_t = torch.cat(hidn_t[-s_len:])[:, 0]
        cls_s = torch.cat(hidn_s)[:, 0]

        cls_s = self.linear(cls_s)

        cls_t = cls_t / torch.norm(cls_t, dim=1, keepdim=True)
        cls_s = cls_s / torch.norm(cls_s, dim=1, keepdim=True)

        loss = (cls_s - cls_t).pow(2).sum(dim=-1).mean()

        return loss





