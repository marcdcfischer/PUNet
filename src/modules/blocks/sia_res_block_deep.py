import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.blocks.sia_block_deep import DeepShiftedInstructedAttentionBlock
from typing import Sequence, Optional, List
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from itertools import chain


class DeepSIAResBlock(nn.Module):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 instruction_pool_size: int,
                 tokens_per_instruction: int,
                 separate_background: bool = True,
                 kernel_size: Sequence[int] = (3, 3, 1),
                 strides: Sequence[int] = (2, 2, 1),
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

        self.act = get_act_layer(name=act)
        self.norm = get_norm_layer(name=norm, spatial_dims=3, channels=self.channels_in)
        self.conv = Convolution(spatial_dims=3,
                                in_channels=self.channels_in,
                                out_channels=self.channels_out,
                                kernel_size=kernel_size,
                                strides=strides,
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

        self.norm_res = get_norm_layer(name=norm, spatial_dims=3, channels=self.channels_in)
        self.conv_res = Convolution(spatial_dims=3,
                                    in_channels=self.channels_in,
                                    out_channels=self.channels_out,
                                    kernel_size=(1, 1, 1),
                                    strides=strides,
                                    act="leakyrelu",
                                    norm="batch",
                                    is_transposed=False,
                                    conv_only=True)

    def forward(self, x: torch.Tensor, x_instructions: Optional[List[torch.Tensor]] = None, label_indices: Optional[torch.Tensor] = None):
        residual = x

        # Downsampling
        x = self.conv(self.act(self.norm(x)))

        # SIA
        x = self.sia(x, x_instructions, label_indices=label_indices)

        # Residuals
        x = x + self.conv_res(self.norm_res(residual))

        return x

    def named_parameters_body(self):

        parameters_down = list(chain(*[self.conv.named_parameters(), self.norm.named_parameters()]))
        parameters_res = list(chain(*[self.conv_res.named_parameters(), self.norm_res.named_parameters()]))
        parameters_sia_att = self.sia.named_parameters_attention()
        parameters_bias_content = self.sia.named_parameters_bias_content()

        return list(chain(*[parameters_down, parameters_res, parameters_sia_att, parameters_bias_content]))

    def named_parameters_adapter(self):

        return self.sia.named_parameters_adapter()

    def named_parameters_bias_instructions(self):

        parameters_bias_instructions = list(self.sia.named_parameters_bias_instructions())

        return parameters_bias_instructions
