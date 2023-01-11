import torch
from torch import nn
from typing import Optional
import einops
from torch.utils.checkpoint import checkpoint


class Attention(nn.Module):
    def __init__(self,
                 q_channels: int,
                 kv_channels: Optional[int] = None,
                 heads_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 heads: int = 4):
        super().__init__()
        kv_channels = kv_channels if kv_channels is not None else q_channels
        heads_channels = heads_channels if heads_channels is not None else q_channels
        out_channels = out_channels if out_channels is not None else q_channels
        inner_channels = heads_channels
        assert inner_channels % heads == 0

        self.scale = heads_channels ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(q_channels, inner_channels, bias=False)
        self.to_k = nn.Linear(kv_channels, inner_channels, bias=False)
        self.to_v = nn.Linear(kv_channels, inner_channels, bias=False)
        self.to_out = nn.Linear(inner_channels, out_channels)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, pos_q: Optional[torch.Tensor] = None, pos_k: Optional[torch.Tensor] = None):
        # Transform to q, k, v and add pos info
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)

        # Rearrange
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Add pos
        q = q + einops.rearrange(pos_q, 'b n d -> b () n d') if pos_q is not None else q
        k = k + einops.rearrange(pos_k, 'b n d -> b () n d') if pos_k is not None else k

        # Attention
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn_score = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn_score, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class WindowedMaskedAttention(nn.Module):
    def __init__(self,
                 q_channels: int,
                 kv_channels: Optional[int] = None,
                 heads_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 separate_norms: bool = False,
                 heads: int = 4,
                 use_checkpoint: bool = True):
        super().__init__()
        kv_channels = kv_channels if kv_channels is not None else q_channels
        heads_channels = heads_channels if heads_channels is not None else q_channels // heads
        out_channels = out_channels if out_channels is not None else q_channels
        inner_channels = q_channels
        assert inner_channels % heads == 0

        self.use_checkpoint = use_checkpoint
        self.heads = heads
        self.eps = 1e-8

        self.pre_norm_q, self.pre_norm_k, self.pre_norm_v = None, None, None
        if separate_norms:
            self.pre_norm_q = nn.LayerNorm(q_channels)
            self.pre_norm_k = nn.LayerNorm(kv_channels)
            self.pre_norm_v = nn.LayerNorm(kv_channels)

        self.to_q = nn.Linear(q_channels, inner_channels, bias=False)
        self.to_k = nn.Linear(kv_channels, inner_channels, bias=False)
        self.to_v = nn.Linear(kv_channels, inner_channels, bias=False)
        self.masked_attention_calc = WindowedMaskedAttentionCalculation(inv_temperature=heads_channels ** -0.5)
        self.to_out = nn.Linear(inner_channels, out_channels)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_q: Optional[torch.Tensor] = None,
                pos_k: Optional[torch.Tensor] = None,
                pos_scores: Optional[torch.Tensor] = None):
        # Transform to q, k, v and add pos info
        q = self.to_q(self.pre_norm_q(q)) if self.pre_norm_q is not None else self.to_q(q)
        k = self.to_k(self.pre_norm_k(k)) if self.pre_norm_k is not None else self.to_k(q)
        v = self.to_v(self.pre_norm_v(v)) if self.pre_norm_v is not None else self.to_v(q)

        # Rearrange
        q, k, v = map(lambda t: einops.rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), (q, k, v))

        # Add pos
        q = q + pos_q if pos_q is not None else q
        k = k + pos_k if pos_k is not None else k

        # Masked Attention
        if self.use_checkpoint:
            out = checkpoint(self.masked_attention_calc, q, k, v, pos_scores, mask, preserve_rng_state=False, use_reentrant=False)
        else:
            out = self.masked_attention_calc(q, k, v, pos_scores, mask)
        out = einops.rearrange(out, 'b p h n d -> b p n (h d)', h=self.heads)
        out = self.to_out(out)

        return out


class WindowedMaskedAttentionCalculation(nn.Module):
    def __init__(self,
                 inv_temperature: float):
        super().__init__()
        self.inv_temperature = inv_temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_scores: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):

        sim = torch.einsum('b p h i d, b p h j d -> b p h i j', q, k)
        sim *= self.inv_temperature     # Scaled prior to pos_scores (since they are already pre-scaled atm)
        sim = sim + pos_scores if pos_scores is not None else sim
        sim = mask * sim if mask is not None else sim
        attn_score = sim.softmax(dim=-1)

        out = torch.einsum('b p h i j, b p h j d -> b p h i d', attn_score, v)

        return out
