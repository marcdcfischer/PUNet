import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Optional, List
from src.modules.blocks.attentive_mechanism_masked import WindowedMaskedAttentionBlock
from src.modules.blocks.query_encodings import DeepInstructedAttentionPositionScores
from einops import rearrange
import math
from itertools import chain


# follows original implementation of swin attention (wmsa, swmsa) loosely:
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py, https://github.com/ahatamiz/MONAI/blob/swin_unetr_v1/monai/networks/nets/swin_unetr.py
class DeepShiftedInstructedAttentionBlock(nn.Module):
    """
    Block consisting of two attention blocks (windowed and shifted & windowed)
    """

    def __init__(self,
                 hidden_channels: int,
                 instruction_pool_size: int,
                 tokens_per_instruction: int,
                 separate_background: bool = True,
                 window_size: Sequence[int] = (8, 8, 1),
                 shift_size: Optional[Sequence[int]] = None,
                 heads: int = 4,
                 unique_instruction_bias: bool = True,
                 unique_token_bias: bool = True,
                 no_bias_instructions: bool = False,
                 no_bias_content: bool = False,
                 adapter: bool = False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.instruction_pool_size = instruction_pool_size
        self.unique_instruction_bias = unique_instruction_bias
        self.unique_token_bias = unique_token_bias
        self.window_size = window_size
        self.shift_size = shift_size if shift_size is not None else tuple(x_ // 2 for x_ in window_size)
        self.heads = heads

        self.msa_blocks = nn.ModuleList()
        self.attention_scores = nn.ModuleList()
        for idx_block in range(2):
            self.msa_blocks.append(WindowedMaskedAttentionBlock(hidden_channels=self.hidden_channels,
                                                                heads=self.heads,
                                                                adapter=adapter))
            self.attention_scores.append(DeepInstructedAttentionPositionScores(embedding_dim=32,  # atm hardcoded
                                                                               heads=self.heads,
                                                                               instruction_pool_size=self.instruction_pool_size,
                                                                               tokens_per_instruction=tokens_per_instruction,
                                                                               separate_background=separate_background,
                                                                               max_absolute_positions=self.window_size,
                                                                               max_capped_distances=self.window_size,
                                                                               unique_instruction_bias=self.unique_instruction_bias,
                                                                               unique_token_bias=self.unique_token_bias,
                                                                               no_bias_instructions=no_bias_instructions,
                                                                               no_bias_content=no_bias_content))

    def get_window_size(self, x_dim):

        window_size_ = list(self.window_size)
        shift_size_ = list(self.shift_size)
        for idx_dim in range(len(x_dim)):
            # Shrink window if whole content is smaller
            if x_dim[idx_dim] <= window_size_[idx_dim]:
                window_size_[idx_dim] = x_dim[idx_dim]
                shift_size_[idx_dim] = 0

        return tuple(window_size_), tuple(shift_size_)

    def get_attn_mask(self,
                      x_shape: Tuple[int, int, int],
                      window_size: Tuple[int, int, int],
                      shift_size: Tuple[int, int, int],
                      paddings: Tuple[int, ...],
                      device: Optional[torch.device] = None):

        with torch.no_grad():
            # calculate attention mask for SW-MSA
            image_mask = torch.zeros(x_shape, dtype=torch.float, device=device)  # [H, W, D]
            h_slices = (slice(0, -window_size[0]),
                        slice(-window_size[0], -shift_size[0]),
                        slice(-shift_size[0], None))
            w_slices = (slice(0, -window_size[1]),
                        slice(-window_size[1], -shift_size[1]),
                        slice(-shift_size[1], None))
            d_slices = (slice(0, -window_size[2]),
                        slice(-window_size[2], -shift_size[2]),
                        slice(-shift_size[2], None))
            # Encode each region by an int
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        image_mask[h, w, d] = cnt
                        cnt += 1

            # Encode non-padded regions differently, so paddings can't interact with true content
            if any([x_ > 0 for x_ in paddings]):
                image_mask[paddings[0]: x_shape[0] - paddings[1],
                           paddings[2]: x_shape[1] - paddings[3],
                           paddings[4]: x_shape[2] - paddings[5]] = 100

            mask_windows = rearrange(self.window_partitioning(rearrange(image_mask, 'h w d -> () () h w d'), window_size).squeeze(dim=2), 'b p h w d -> b p (h w d)')  # [1 (B), P, (H' W' D')]
            attn_mask = rearrange(mask_windows, 'b p n -> b p () n') - rearrange(mask_windows, 'b p n -> b p n ()')  # [1 (B), P, (H' W' D'), (H' W' D')]
            attn_mask = (~(attn_mask == 0)).float()  # Multiplicative mask with zeros for regions with different int encoding

        return attn_mask

    @staticmethod
    def window_partitioning(x: torch.Tensor, window_size: Tuple[int, int, int]):
        """
        :param x: [B, C, H, W, D]. Expects image content (not nodes [B N C])
        :return: [B, P, C, H', W', D']
        """

        x = rearrange(x, 'b c (h p1) (w p2) (d p3) -> b (p1 p2 p3) c h w d', h=window_size[0], w=window_size[1], d=window_size[2])

        return x

    @staticmethod
    def window_recombination(x: torch.Tensor, window_size: Tuple[int, int, int], x_shape: Tuple[int, int, int]):
        """
        :param x: [B, P, C, H', W', D']. Expects image content (not nodes [B N C])
        :return: [B, C, H, W, D]
        """

        x = rearrange(x, 'b (p1 p2 p3) c h w d -> b c (h p1) (w p2) (d p3)', p1=x_shape[0] // window_size[0], p2=x_shape[1] // window_size[1], p3=x_shape[2] // window_size[2])

        return x

    def forward(self, x: torch.Tensor, x_instructions: Optional[List[torch.Tensor]] = None, label_indices: Optional[torch.Tensor] = None):
        """
        :param x: [B, C, H, W, D]. Expects image content (not nodes [B N C])
        :param x_instructions: [B, I, C]
        :return: [B, C, H, W, D], [B, I, C]
        """
        b, c, h, w, d = x.shape
        n_instructions = x_instructions[0].shape[1] if x_instructions is not None else 0
        window_size_, shift_size_ = self.get_window_size((h, w, d))

        # Pad content if necessary
        paddings = (0, 0, 0, 0, 0, 0)
        if any([h % window_size_[0] != 0, w % window_size_[1], d % window_size_[2]]):
            paddings = [math.floor((window_size_[0] - h % window_size_[0]) / 2), math.ceil((window_size_[0] - h % window_size_[0]) / 2),
                        math.floor((window_size_[1] - w % window_size_[1]) / 2), math.ceil((window_size_[1] - w % window_size_[1]) / 2),
                        math.floor((window_size_[2] - d % window_size_[2]) / 2), math.ceil((window_size_[2] - d % window_size_[2]) / 2)]
            paddings[-1] = 0 if window_size_[2] == 1 else paddings[-1]  # don't pad depth singleton dim.
            x = F.pad(x, tuple(reversed(paddings)))  # F.pad needs reverse order (starting from last)
        h_padded, w_padded, d_padded = x.shape[2:]

        # Calc mask attention for cyclically shifted content
        with torch.no_grad():
            attn_mask_ = self.get_attn_mask(x_shape=(h_padded, w_padded, d_padded), window_size=window_size_, shift_size=shift_size_, paddings=paddings, device=x.device)  # [1 (B), P, (H W D), (H W D)]. For cyclically shifted elements in shifted attention
            attn_mask_all = torch.zeros((attn_mask_.shape[0], attn_mask_.shape[1], n_instructions + attn_mask_.shape[2], n_instructions + attn_mask_.shape[3]), dtype=torch.float, device=x.device)  # [1 (B), P, I + (H W D), I + (H W D)]. Instructions are never masked.
            attn_mask_all[:, :, n_instructions:, n_instructions:] = attn_mask_  # mask for content -> content interaction
            if x_instructions is not None:
                # Enable instruction -> content interaction if instructions are given.
                attn_mask_all[:, :, n_instructions:, :n_instructions] = 1.  # mask for instruction -> content interaction. (lazy way of excluding inst -> inst, content -> inst)
            attn_mask_all = rearrange(attn_mask_all, 'b p n m -> b p () n m')  # [1 (B), P, 1 (H), N, N]

        # WMSA
        # Calc position-based scores (including those for instructions, content and cross-attention)
        pos_scores = self.attention_scores[0](dim_q=n_instructions + math.prod(window_size_), dim_k=n_instructions + math.prod(window_size_), dim_i=n_instructions,
                                              dim_h=window_size_[0], dim_w=window_size_[1], dim_d=window_size_[2], label_indices=label_indices, device=x.device)  # [1 (B), H, I + (H W D), I + (H W D)]
        pos_scores = rearrange(pos_scores, 'b h n m -> b () h n m')  # [B, 1 (P), H, N, N]

        x_windowed = self.window_partitioning(x, window_size_)  # [B', P, C, H', W', D']
        if x_instructions is not None:
            x_all = torch.cat([rearrange(x_instructions[0], 'b i c -> b () i c').expand(-1, x_windowed.shape[1], -1, -1),
                               rearrange(x_windowed, 'b p c h w d -> b p (h w d) c')], dim=2)  # Concat instructions
        else:
            x_all = rearrange(x_windowed, 'b p c h w d -> b p (h w d) c')
        x_all = self.msa_blocks[0](x_all, pos_scores=pos_scores)
        # x_instructions = x_all[:, :, :n_instructions, :].mean(dim=1) if x_instructions is not None else x_instructions  # For instruction take average of all windows
        x_windowed = rearrange(x_all[:, :, n_instructions:, :], 'b p (h w d) c -> b p c h w d', h=window_size_[0], w=window_size_[1], d=window_size_[2])
        x = self.window_recombination(x_windowed, window_size_, x_shape=(h_padded, w_padded, d_padded))

        # SWMSA
        # Calc position-based scores (including those for instructions, content and cross-attention)
        pos_scores = self.attention_scores[1](dim_q=n_instructions + math.prod(window_size_), dim_k=n_instructions + math.prod(window_size_), dim_i=n_instructions,
                                              dim_h=window_size_[0], dim_w=window_size_[1], dim_d=window_size_[2], label_indices=label_indices, device=x.device)  # [1 (B), H, I + (H W D), I + (H W D)]
        pos_scores = rearrange(pos_scores, 'b h n m -> b () h n m')  # [B, 1 (P), H, N, N]

        x = torch.roll(x, shifts=(-shift_size_[0], -shift_size_[1], -shift_size_[2]), dims=(2, 3, 4)) if any([x_ > 0 for x_ in shift_size_]) else x  # shifted backwards
        x_windowed = self.window_partitioning(x, window_size_)  # [B', P, C, H', W', D']
        if x_instructions is not None:
            x_all = torch.cat([rearrange(x_instructions[1], 'b i c -> b () i c').expand(-1, x_windowed.shape[1], -1, -1),
                               rearrange(x_windowed, 'b p c h w d -> b p (h w d) c')], dim=2)  # Concat instructions
        else:
            x_all = rearrange(x_windowed, 'b p c h w d -> b p (h w d) c')
        x_all = self.msa_blocks[1](x_all, mask=attn_mask_all, pos_scores=pos_scores)
        # x_instructions = x_all[:, :, :n_instructions, :].mean(dim=1) if x_instructions is not None else x_instructions  # For instruction take average of all windows
        x_windowed = rearrange(x_all[:, :, n_instructions:, :], 'b p (h w d) c -> b p c h w d', h=window_size_[0], w=window_size_[1], d=window_size_[2])
        x = self.window_recombination(x_windowed, window_size_, x_shape=(h_padded, w_padded, d_padded))
        x = torch.roll(x, shifts=shift_size_, dims=(2, 3, 4)) if any([x_ > 0 for x_ in shift_size_]) else x  # shifted forwards (reverse)

        # Crop padded content if necessary
        if any([x_ > 0 for x_ in paddings]):
            x = x[...,
                  paddings[0]: x.shape[2] - paddings[1],
                  paddings[2]: x.shape[3] - paddings[3],
                  paddings[4]: x.shape[4] - paddings[5]]  # F.pad needs reverse order (starting from last)

        return x

    def named_parameters_attention(self):

        return list(chain(*[x_.named_parameters_body() for x_ in self.msa_blocks]))

    def named_parameters_adapter(self):

        return list(chain(*[x_.named_parameters_adapter() for x_ in self.msa_blocks]))

    def named_parameters_bias_content(self):

        return list(chain(*[x_.named_parameters_bias_content() for x_ in self.attention_scores]))

    def named_parameters_bias_instructions(self):

        return list(chain(*[x_.named_parameters_bias_instructions() for x_ in self.attention_scores]))
