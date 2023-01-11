import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.blocks.sia_block_deep import DeepShiftedInstructedAttentionBlock
from typing import Sequence, Optional, List
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from itertools import chain


class DeepSIAUpBlock(nn.Module):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 instruction_pool_size: int,
                 tokens_per_instruction: int,
                 separate_background: bool = True,
                 kernel_size: Sequence[int] = (3, 3, 1),
                 strides: Sequence[float] = (2, 2, 1),
                 heads: int = 4,
                 window_size: Sequence[int] = (8, 8, 1),
                 act: str = "leakyrelu",
                 norm: str = "batch",
                 unique_instruction_bias: bool = True,
                 unique_token_bias: bool = True,
                 no_bias_instructions: bool = False,
                 no_bias_content: bool = False,
                 adapter: bool = False):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.up = nn.Upsample(scale_factor=strides, mode='trilinear', align_corners=False)

        self.act = get_act_layer(name=act)
        self.norm_concat = get_norm_layer(name=norm, spatial_dims=3, channels=self.channels_in + self.channels_out)
        self.conv_concat = Convolution(spatial_dims=3,
                                       in_channels=self.channels_in + self.channels_out,
                                       out_channels=self.channels_out,
                                       kernel_size=kernel_size,
                                       strides=(1, 1, 1),
                                       act="leakyrelu",
                                       norm="batch",
                                       is_transposed=False,
                                       conv_only=True)

        self.sia = DeepShiftedInstructedAttentionBlock(hidden_channels=self.channels_out,
                                                       instruction_pool_size=instruction_pool_size,
                                                       tokens_per_instruction=tokens_per_instruction,
                                                       separate_background=separate_background,
                                                       heads=heads,
                                                       window_size=window_size,
                                                       unique_instruction_bias=unique_instruction_bias,
                                                       unique_token_bias=unique_token_bias,
                                                       no_bias_instructions=no_bias_instructions,
                                                       no_bias_content=no_bias_content,
                                                       adapter=adapter)

    def forward(self, x: torch.Tensor, x_skips: torch.Tensor, x_instructions: Optional[List[torch.Tensor]] = None, label_indices: Optional[torch.Tensor] = None):

        # Upsample lower res content
        x = self.up(x)

        # Concat and linear / conv projection of channels
        assert all([x_ - y_ <= 1 for x_, y_ in zip(x.shape[2:], x_skips.shape[2:])])
        x = self.conv_concat(self.act(self.norm_concat(torch.cat([x[...,
                                                                    :x_skips.shape[2],
                                                                    :x_skips.shape[3],
                                                                    :x_skips.shape[4]], x_skips], dim=1))))  # cropped if upsampled version is too large.

        # SIA
        x = self.sia(x, x_instructions, label_indices=label_indices)

        return x

    def named_parameters_body(self):

        parameters_down = list(chain(*[self.conv_concat.named_parameters(), self.norm_concat.named_parameters()]))
        parameters_sia_att = self.sia.named_parameters_attention()
        parameters_bias_content = self.sia.named_parameters_bias_content()

        return list(chain(*[parameters_down, parameters_sia_att, parameters_bias_content]))

    def named_parameters_adapter(self):

        return self.sia.named_parameters_adapter()

    def named_parameters_bias_instructions(self):

        parameters_bias_instructions = list(self.sia.named_parameters_bias_instructions())

        return parameters_bias_instructions
