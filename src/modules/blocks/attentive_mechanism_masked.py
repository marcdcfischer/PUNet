import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.modules.blocks.attention import WindowedMaskedAttention
from torch.utils.checkpoint import checkpoint


# see https://github.com/KMnP/vpt/blob/e2dd70a5ee291d398d002e6963ddbe0f66f58038/src/models/vit_adapter/adapter_block.py#L25 for adapter adaptation
class WindowedMaskedAttentionBlock(nn.Module):
    """
    Generic attention block for masked attention
    """
    def __init__(self,
                 hidden_channels: int,
                 heads: int = 4,
                 reduction_factor: int = 4,  # atm hardcoded
                 adapter: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()
        self.adapter = adapter
        self.use_checkpoint = use_checkpoint

        self.attention_norm = nn.LayerNorm(hidden_channels)
        self.attention = WindowedMaskedAttention(q_channels=hidden_channels,
                                                 heads=heads,
                                                 separate_norms=False)
        self.mlp_norm = nn.LayerNorm(hidden_channels)
        self.mlp = nn.Linear(hidden_channels, hidden_channels)

        if self.adapter:
            self.linear_down = nn.Linear(hidden_channels, hidden_channels // reduction_factor)
            self.linear_up = nn.Linear(hidden_channels // reduction_factor, hidden_channels)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, pos_scores: Optional[torch.Tensor] = None):
        """

        :param x: Key / value content [B, P, N, C]
        :return:
        """

        # Attention block
        x_att = self.attention_norm(x)
        if self.use_checkpoint:
            x_att = checkpoint(self.attention, x_att, x_att, x_att, mask, None, None, pos_scores, preserve_rng_state=False, use_reentrant=False)
        else:
            x_att = self.attention(q=x_att, k=x_att, v=x_att, mask=mask, pos_scores=pos_scores)
        x = x + x_att

        # Residual and MLP
        x_mlp = self.mlp_norm(x)  # For next iteration add a small MLP with at least on activation
        x_mlp = self.mlp(x_mlp)
        x = x + x_mlp

        # Adapter - with pre- and post- residual connection
        if self.adapter:
            x = x + self.linear_up(F.leaky_relu(self.linear_down(x)))

        return x

    def named_parameters_body(self):

        return [*list(self.attention_norm.named_parameters()), *list(self.attention.named_parameters()),
                *list(self.mlp_norm.named_parameters()), *list(self.mlp.named_parameters())]

    def named_parameters_adapter(self):

        params_ = list()
        if self.adapter:
            params_ += [*self.linear_down.named_parameters(), *self.linear_up.named_parameters()]

        return params_

