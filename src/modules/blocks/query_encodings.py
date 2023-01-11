import torch
import torch.nn as nn
import einops
import math
from typing import Tuple, Optional, Sequence
import warnings


# see e.g. https://github.com/lucidrains/x-transformers/blob/55ca5d96c8b850b064177091f7a1dcfe784b24ce/x_transformers/x_transformers.py#L116
# Note: Perceiver uses slightly different formulation based on range from -1 to 1.
# Explanation - https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
class FourierPositionalEncoding(nn.Module):
    """ Generate "classical" fourier (sinusoidal) pos encoding """
    def __init__(self,
                 pos_channels: int = 64,
                 dims: int = 1,
                 base: float = 10000.):
        super().__init__()
        self.pos_channels = pos_channels
        self.dims = dims
        self.pos_channels_per_dim = self.pos_channels // self.dims
        inv_freq = torch.exp(- torch.arange(0., self.pos_channels_per_dim, 2) / self.pos_channels_per_dim * math.log(base))  # base**(-2k/ch) = 1 / base**(2k/ch), k = 0, 2*1, ..., pos_channels_per_dim - 2
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, dims: Tuple[int] = (2, 3, 4), offset: Tuple[int] = (0, 0, 0)):
        emb_last = None
        for idx_dim, dim_ in enumerate(dims):
            pos_dim_ = torch.arange(0., x.shape[dim_], device=x.device) + offset[idx_dim]
            sinusoid_inp_ = torch.einsum('i,j->ij', pos_dim_, self.inv_freq)
            emb_ = torch.cat([sinusoid_inp_.sin(), sinusoid_inp_.cos()], dim=-1)
            if idx_dim > 0:
                emb_last = einops.repeat(emb_last, 'm i -> m n i', n=emb_.shape[0])
                emb_ = einops.repeat(emb_, 'n j -> m n j', m=emb_last.shape[0])
                emb_ = einops.rearrange(torch.concat([emb_last, emb_], dim=-1), 'm n k -> (m n) k')
            emb_last = emb_
        # Fill remaining dims with zeros
        if emb_last.shape[1] < self.pos_channels:
            emb_last = torch.cat([emb_last, torch.zeros((emb_last.shape[0], self.pos_channels - emb_last.shape[1]), device=emb_last.device)], dim=-1)
        return einops.rearrange(emb_last, 'n d -> () n d')


class SparseFourierPositionalEncoding(nn.Module):
    """ Generate "classical" fourier (sinusoidal) pos encoding """
    def __init__(self,
                 pos_channels: int = 64,
                 dims: int = 1,
                 base: float = 10000.):
        super().__init__()
        self.pos_channels = pos_channels
        self.dims = dims
        self.pos_channels_per_dim = self.pos_channels // self.dims
        inv_freq = torch.exp(- torch.arange(0., self.pos_channels_per_dim, 2) / self.pos_channels_per_dim * math.log(base))  # base**(-2k/ch) = 1 / base**(2k/ch), k = 0, 2*1, ..., pos_channels_per_dim - 2
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q: torch.Tensor, x_shape: Tuple[int], offset: Tuple[int] = (0, 0, 0)):
        """
        :param q: [B N 3]. Positions from -1 to 1 across each dimension
        :param x_shape: Shape of x. (h, w, d)
        :param offset:
        :return:
        """

        emb_last = None
        for idx_dim in range(q.shape[-1]):
            pos_dim_ = (q[..., idx_dim] + 1. / 2.) * (x_shape[idx_dim] - 1.) + offset[idx_dim]
            sinusoid_inp_ = torch.einsum('bi,j->bij', pos_dim_, self.inv_freq)
            emb_ = torch.cat([sinusoid_inp_.sin(), sinusoid_inp_.cos()], dim=-1)
            if idx_dim > 0:
                emb_ = torch.concat([emb_last, emb_], dim=-1)
            emb_last = emb_
        # Fill remaining dims with zeros
        if emb_last.shape[2] < self.pos_channels:
            emb_last = torch.cat([emb_last, torch.zeros((emb_last.shape[0], emb_last.shape[1], self.pos_channels - emb_last.shape[2]), device=emb_last.device)], dim=-1)
        return emb_last


class LearnedQuery(nn.Module):
    def __init__(self,
                 n_queries: int,
                 query_channels: int = 512,
                 requires_grad: bool = True):
        super().__init__()
        self.query = nn.Parameter(nn.init.xavier_uniform_(torch.empty((n_queries, query_channels)),
                                                          gain=nn.init.calculate_gain('linear')), requires_grad=requires_grad)  # [N, C]. Perceiver used truncated normal

    def forward(self, batch_size: int):
        return einops.repeat(self.query, 'n c -> b n c', b=batch_size)


class LearnedNormedQuery(nn.Module):
    def __init__(self,
                 n_queries: int,
                 query_channels: int = 512,
                 requires_grad: bool = True):
        super().__init__()
        self.query = nn.Parameter(nn.init.xavier_uniform_(torch.empty((n_queries, query_channels)),
                                                          gain=nn.init.calculate_gain('linear')), requires_grad=requires_grad)  # [N, C]. Perceiver used truncated normal
        self.query_norm = nn.LayerNorm(query_channels, elementwise_affine=True)

    def forward(self, batch_size: int):
        return einops.repeat(self.query_norm(self.query), 'n c -> b n c', b=batch_size)


class LearnedNormedInstruction(nn.Module):
    """ List of learned embeddings vector """
    def __init__(self,
                 instruction_pool_size: int,
                 tokens_per_instruction: int = 10,
                 instruction_channels: int = 512,
                 requires_grad: bool = True,
                 use_norm: bool = False,
                 elementwise_affine: bool = True):
        super().__init__()
        self.instruction_pool_size = instruction_pool_size
        self.instructions = nn.ParameterList()
        for idx_i in range(instruction_pool_size):
            self.instructions.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty((tokens_per_instruction, instruction_channels)),
                                                                          gain=nn.init.calculate_gain('linear')), requires_grad=requires_grad))  # [N, C]
        self.instructions_norm = nn.LayerNorm(instruction_channels, elementwise_affine=elementwise_affine) if use_norm else None  # Atm joint norm for all instructions.

    def forward(self):

        instructions = torch.stack(list(self.instructions), dim=0)
        instructions = self.instructions_norm(instructions) if self.instructions_norm is not None else instructions

        return instructions  # [I, N, C]


class LearnedNormedPseudoInstruction(nn.Module):
    """ List of learned embeddings vector """
    def __init__(self,
                 instruction_pool_size_subjects: int,
                 instruction_pool_size_labels: int,
                 tokens_per_instruction: int = 10,
                 instruction_channels: int = 512,
                 requires_grad: bool = True,
                 elementwise_affine: bool = True):
        super().__init__()
        self.instruction_pool_size_subjects = instruction_pool_size_subjects
        self.instruction_pool_size_labels = instruction_pool_size_labels
        self.instructions = nn.ParameterList()
        for idx_i in range(instruction_pool_size_subjects):
            self.instructions.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty((instruction_pool_size_labels, tokens_per_instruction, instruction_channels)),
                                                                          gain=nn.init.calculate_gain('linear')), requires_grad=requires_grad))  # [N, C]
        self.instructions_norm = nn.LayerNorm(instruction_channels, elementwise_affine=elementwise_affine)  # Atm joint norm for all instructions.

    def forward(self, idx_subject, idx_label):

        # Gather and update only params of available subjects (to prevent potential excessive grad calc)
        instructions = self.instructions_norm(self.instructions[idx_subject][idx_label, ...])

        return instructions  # [N, C]


# See https://github.com/lucidrains/x-transformers/blob/55ca5d96c8b850b064177091f7a1dcfe784b24ce/x_transformers/x_transformers.py#L116 for potential implementations
# https://github.com/epfml/attention-cnn/blob/master/models/bert.py
# https://github.com/TensorUI/relative-position-pytorch/blob/master/relative_position.py
class InstructedAttentionPositionScores(nn.Module):
    """
     A combination of neural interpreter learned position, alibi (https://arxiv.org/pdf/2108.12409.pdf), relative position representations (https://arxiv.org/pdf/1803.02155.pdf, https://arxiv.org/pdf/1906.05909.pdf) and other relative position schemes.
     A (computational) efficient form for an attention score: q^T * k + w_h^T * emb[diff] with a learned embedding for each diff and a learned weight vector for each category and attention head.
    """
    def __init__(self,
                 embedding_dim: int = 64,
                 heads: int = 4,
                 tokens_per_instruction: int = 10,
                 max_absolute_positions: Sequence[int] = (64, 64, 1),  # Max absolute positions index
                 max_capped_distances: Sequence[int] = (64, 64, 1),  # Max capped relative distances
                 unique_token_bias: bool = True,
                 no_bias_instructions: bool = False,  # Disables weights for instructions and cross biases.
                 no_bias_content: bool = False):  # Disables content weights
        super().__init__()
        self.heads = heads
        self.tokens_per_instruction = tokens_per_instruction
        self.unique_token_bias = unique_token_bias
        self.max_token_positions = tokens_per_instruction if unique_token_bias else 1
        self.max_absolute_positions = max_absolute_positions
        self.max_capped_distances = max_capped_distances
        self.no_bias_instructions = no_bias_instructions
        self.no_bias_content = no_bias_content
        self.inv_temperature = embedding_dim ** -0.5

        # Learned encoding
        self.encoding_intra_instructions = nn.Parameter(nn.init.xavier_uniform_(torch.empty((self.max_token_positions, self.max_token_positions, embedding_dim)),
                                                                                gain=nn.init.calculate_gain('linear')), requires_grad=True)  # Encoding for instructions of (intra) connections same category
        self.encoding_inter_instructions = nn.Parameter(nn.init.xavier_uniform_(torch.empty((self.max_token_positions, self.max_token_positions, embedding_dim)),
                                                                                gain=nn.init.calculate_gain('linear')), requires_grad=True)  # Encoding for instructions of (inter) connections of different categories
        self.encoding_cross_inst_content = nn.Parameter(nn.init.xavier_uniform_(torch.empty((self.max_token_positions, 1, embedding_dim)),
                                                                                gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_cross_content_inst = nn.Parameter(nn.init.xavier_uniform_(torch.empty((1, self.max_token_positions, embedding_dim)),
                                                                                gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_content_h = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[0] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_content_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[1] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_content_d = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[2] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)

        # To be encountered relative distances
        relative_distances_h = torch.arange(self.max_absolute_positions[0], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[0], dtype=torch.long).reshape(-1, 1)
        relative_distances_h = torch.clamp(relative_distances_h + self.max_capped_distances[0] - 1, min=0, max=(self.max_capped_distances[0] - 1) * 2)
        self.register_buffer('relative_distances_h', relative_distances_h)
        relative_distances_w = torch.arange(self.max_absolute_positions[1], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[1], dtype=torch.long).reshape(-1, 1)
        relative_distances_w = torch.clamp(relative_distances_w + self.max_capped_distances[1] - 1, min=0, max=(self.max_capped_distances[1] - 1) * 2)
        self.register_buffer('relative_distances_w', relative_distances_w)
        relative_distances_d = torch.arange(self.max_absolute_positions[2], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[2], dtype=torch.long).reshape(-1, 1)
        relative_distances_d = torch.clamp(relative_distances_d + self.max_capped_distances[2] - 1, min=0, max=(self.max_capped_distances[2] - 1) * 2)
        self.register_buffer('relative_distances_d', relative_distances_d)

        # Learned weights (per head) to calculate score - similar to neural interpreter
        # Note: this variant is query independent (this replaces q in q^T * emb[diff]).
        self.weights_intra_instructions = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                               gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_inter_instructions = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                               gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_cross_inst_content = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                               gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_cross_content_inst = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                               gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_h = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_d = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)

    def forward(self, dim_q, dim_k, dim_i, dim_h, dim_w, dim_d, device: Optional[torch.device] = None):
        """
        :param dim_q: queries dim
        :param dim_k: keys dim
        :param dim_i: actual instructions dim
        :param dim_h: actual height dim
        :param dim_w: actual width dim
        :param dim_d: actual depth dim
        :return: additive attention scores
        """

        # Retrieve embeddings according to relative / absolute / categorical position
        n_instruction_categories = dim_i // self.tokens_per_instruction
        if dim_i > 0:
            assert n_instruction_categories > 0
        intra_mask = torch.eye(n_instruction_categories, dtype=torch.float, device=device).reshape(n_instruction_categories, 1, n_instruction_categories, 1, 1) \
            .expand(-1, self.tokens_per_instruction, -1, self.tokens_per_instruction, -1).reshape(dim_i, dim_i, 1)  # Block sparse mask [I, I, 1]
        if self.unique_token_bias:
            # Unique learnable positional embedding for all tokens (across all instructions)
            intra_instruction_embeddings = intra_mask * self.encoding_intra_instructions.repeat(n_instruction_categories, n_instruction_categories, 1)  # [I, I, C]
            inter_instruction_embeddings = (~intra_mask.bool()).float() * self.encoding_inter_instructions.repeat(n_instruction_categories, n_instruction_categories, 1)  # [I, I, C]
            cross_inst_content_embeddings = self.encoding_cross_inst_content.repeat(n_instruction_categories, 1, 1)
            cross_content_inst_embeddings = self.encoding_cross_content_inst.repeat(1, n_instruction_categories, 1)
        else:
            # Same learnable positional embedding for all tokens (across all instructions)
            intra_instruction_embeddings = intra_mask * self.encoding_intra_instructions.expand(dim_i, dim_i, -1)  # [I, I, C]
            inter_instruction_embeddings = (~intra_mask.bool()).float() * self.encoding_inter_instructions.expand(dim_i, dim_i, -1)  # [I, I, C]
            cross_inst_content_embeddings = self.encoding_cross_inst_content.expand(dim_i, -1, -1)
            cross_content_inst_embeddings = self.encoding_cross_content_inst.expand(-1, dim_i, -1)

        row_embeddings = self.encoding_content_h[self.relative_distances_h[:dim_h, :dim_h], :]  # [H, H, C]. Relative row positions
        col_embeddings = self.encoding_content_w[self.relative_distances_w[:dim_w, :dim_w], :]  # [W, W, C]. Relative column positions
        depth_embeddings = self.encoding_content_d[self.relative_distances_d[:dim_d, :dim_d], :]  # [D, D, C]. Relative depth positions

        intra_instruction_scores = torch.einsum('h c, n m c -> h n m', self.weights_intra_instructions, intra_instruction_embeddings)  # [Heads, I, I]
        inter_instruction_scores = torch.einsum('h c, n m c -> h n m', self.weights_inter_instructions, inter_instruction_embeddings)  # [Heads, I, I]
        instruction_scores = intra_instruction_scores + inter_instruction_scores
        cross_inst_content_scores = torch.einsum('h c, n m c -> h n m', self.weights_cross_inst_content, cross_inst_content_embeddings)  # [Heads, I, 1]
        cross_content_inst_scores = torch.einsum('h c, n m c -> h n m', self.weights_cross_content_inst, cross_content_inst_embeddings)  # [Heads, 1, I]
        row_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_h, row_embeddings)  # [Heads, H, H]
        col_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_w, col_embeddings)  # [Heads, W, W]
        depth_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_d, depth_embeddings)  # [Heads, D, D]
        content_scores = einops.rearrange(row_scores, 'h n m -> h n () () m () ()') + einops.rearrange(col_scores, 'h n m -> h () n () () m ()') + einops.rearrange(depth_scores, 'h n m -> h () () n () () m')  # [Heads, H, W, D, H, W, D]
        content_scores /= 3
        content_scores = einops.rearrange(content_scores, 'h i j k l m n -> h (i j k) (l m n)')  # [Heads, #Content, #Content]

        # Attention score matrix
        # A | B
        # - - -
        # C | D
        scores = torch.zeros((self.heads, dim_q, dim_k), dtype=torch.float, device=device)
        if not self.no_bias_instructions:
            scores[:, :dim_i, :dim_i] = instruction_scores  # [Heads, I, I]. Matrix A
            scores[:, :dim_i, dim_i:] = cross_inst_content_scores.expand(-1, -1, dim_k - dim_i)  # [Heads, I, #Content]. Matrix B
            scores[:, dim_i:, :dim_i] = cross_content_inst_scores.expand(-1, dim_q - dim_i, -1)  # [Heads, #Content, I]. Matrix C
        if not self.no_bias_content:
            scores[:, dim_i:, dim_i:] = content_scores  # [Heads, #Content, #Content]. Matrix D

        # (Pre-)scale scores
        scores *= self.inv_temperature

        return einops.rearrange(scores, 'h q k -> () h q k')

    def parameters_bias_content(self):

        return [self.encoding_content_h, self.encoding_content_w, self.encoding_content_d,
                self.weights_content_h, self.weights_content_w, self.weights_content_d]

    def parameters_bias_instructions(self):

        return [self.encoding_intra_instructions, self.encoding_inter_instructions, self.encoding_cross_content_inst, self.encoding_cross_inst_content,
                self.weights_intra_instructions, self.weights_inter_instructions, self.weights_cross_content_inst, self.weights_cross_inst_content]


class DeepInstructedAttentionPositionScores(nn.Module):
    """
     Only inst -> cont and relative positions are needed for this case (others are 0 since instructions are not further used).
    """
    def __init__(self,
                 embedding_dim: int = 64,
                 heads: int = 4,
                 instruction_pool_size: int = 2,
                 tokens_per_instruction: int = 10,
                 separate_background: bool = True,
                 max_absolute_positions: Sequence[int] = (64, 64, 1),  # Max absolute positions index
                 max_capped_distances: Sequence[int] = (64, 64, 1),  # Max capped relative distances
                 unique_instruction_bias: bool = True,
                 unique_token_bias: bool = True,
                 no_bias_instructions: bool = False,  # Disables weights for instructions and cross biases.
                 no_bias_content: bool = False):  # Disables content weights
        super().__init__()
        self.heads = heads
        self.tokens_per_instruction = tokens_per_instruction
        self.separate_background = separate_background
        self.unique_instruction_bias = unique_instruction_bias
        self.unique_token_bias = unique_token_bias
        self.max_instructions = instruction_pool_size if unique_instruction_bias else 1
        self.max_token_positions = tokens_per_instruction if unique_token_bias else 1
        self.max_absolute_positions = max_absolute_positions
        self.max_capped_distances = max_capped_distances
        self.no_bias_instructions = no_bias_instructions
        self.no_bias_content = no_bias_content
        self.embedding_dim = embedding_dim
        self.inv_temperature = embedding_dim ** -0.5

        # Learned encoding
        self.encoding_cross_inst_content = nn.ParameterList()
        for _ in range(self.max_instructions):
            self.encoding_cross_inst_content.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty((self.max_token_positions, embedding_dim)),
                                                                 gain=nn.init.calculate_gain('linear')), requires_grad=True))
        self.encoding_content_h = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[0] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_content_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[1] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_content_d = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[2] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)

        # To be encountered relative distances
        relative_distances_h = torch.arange(self.max_absolute_positions[0], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[0], dtype=torch.long).reshape(-1, 1)
        relative_distances_h = torch.clamp(relative_distances_h + self.max_capped_distances[0] - 1, min=0, max=(self.max_capped_distances[0] - 1) * 2)
        self.register_buffer('relative_distances_h', relative_distances_h)
        relative_distances_w = torch.arange(self.max_absolute_positions[1], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[1], dtype=torch.long).reshape(-1, 1)
        relative_distances_w = torch.clamp(relative_distances_w + self.max_capped_distances[1] - 1, min=0, max=(self.max_capped_distances[1] - 1) * 2)
        self.register_buffer('relative_distances_w', relative_distances_w)
        relative_distances_d = torch.arange(self.max_absolute_positions[2], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[2], dtype=torch.long).reshape(-1, 1)
        relative_distances_d = torch.clamp(relative_distances_d + self.max_capped_distances[2] - 1, min=0, max=(self.max_capped_distances[2] - 1) * 2)
        self.register_buffer('relative_distances_d', relative_distances_d)

        # Learned weights (per head) to calculate score - similar to neural interpreter
        # Note: this variant is query independent (this replaces q in q^T * emb[diff]).
        self.weights_cross_inst_content = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                               gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_h = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_d = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)

    def forward(self, dim_q, dim_k, dim_i, dim_h, dim_w, dim_d, label_indices: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        """
        :param dim_q: queries dim
        :param dim_k: keys dim
        :param dim_i: actual instructions dim
        :param dim_h: actual height dim
        :param dim_w: actual width dim
        :param dim_d: actual depth dim
        :return: additive attention scores
        """

        # Retrieve embeddings according to relative / absolute / categorical position
        n_instruction_categories = dim_i // self.tokens_per_instruction
        if dim_i > 0:
            assert n_instruction_categories > 0

            if label_indices is not None:
                encodings_ = list()
                for idx_batch in range(label_indices.shape[0]):
                    if self.separate_background:
                        instruction_indices_true = torch.nonzero(label_indices[idx_batch, 1:], as_tuple=False).squeeze(dim=1) + 1  # All indices with designated background instruction ignored
                    else:
                        instruction_indices_true = torch.nonzero(label_indices[idx_batch, :], as_tuple=False).squeeze(dim=1)  # All indices including shared background
                    encoding_ = torch.concat(list(self.encoding_cross_inst_content[instruction_indices_true]), dim=0) if instruction_indices_true.numel() > 1 else self.encoding_cross_inst_content[instruction_indices_true]
                    encodings_.append(encoding_)  # [I_active * T,  C]
                encodings_ = torch.stack(encodings_, dim=0)  # [B, I_active * T, C]
            else:
                encodings_ = torch.concat(list(self.encoding_cross_inst_content[:n_instruction_categories]), dim=0).unsqueeze(dim=0) if n_instruction_categories > 1 else self.encoding_cross_inst_content[:n_instruction_categories] # [1, I_active * T, C]
            if self.unique_instruction_bias and self.unique_token_bias:
                # Unique learnable positional embedding for all tokens and all instructions
                cross_inst_content_embeddings = encodings_
            elif not self.unique_instruction_bias and self.unique_token_bias:
                cross_inst_content_embeddings = self.encoding_cross_inst_content[0].unsqueeze(dim=0).repeat(1, n_instruction_categories, 1)
            elif self.unique_instruction_bias and not self.unique_token_bias:
                cross_inst_content_embeddings = encodings_.repeat_interleave(1, self.tokens_per_instruction, 1)
            else:
                # Same learnable positional embedding for all tokens (across all instructions)
                cross_inst_content_embeddings = self.encoding_cross_inst_content[0].unsqueeze(dim=0).expand(-1, dim_i, -1)
        else:
            warnings.warn(f'Using empty bias score.')
            cross_inst_content_embeddings = torch.zeros((1, dim_i, self.embedding_dim), dtype=torch.float, device=device)

        row_embeddings = self.encoding_content_h[self.relative_distances_h[:dim_h, :dim_h], :]  # [H, H, C]. Relative row positions
        col_embeddings = self.encoding_content_w[self.relative_distances_w[:dim_w, :dim_w], :]  # [W, W, C]. Relative column positions
        depth_embeddings = self.encoding_content_d[self.relative_distances_d[:dim_d, :dim_d], :]  # [D, D, C]. Relative depth positions

        cross_inst_content_scores = torch.einsum('h c, b n c -> b h n', self.weights_cross_inst_content, cross_inst_content_embeddings)  # [B, Heads, I]
        row_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_h, row_embeddings).unsqueeze(dim=0)  # [1, Heads, H, H]
        col_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_w, col_embeddings).unsqueeze(dim=0)  # [1, Heads, W, W]
        depth_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_d, depth_embeddings).unsqueeze(dim=0)  # [1, Heads, D, D]
        content_scores = einops.rearrange(row_scores, 'b h n m -> b h n () () m () ()') + einops.rearrange(col_scores, 'b h n m -> b h () n () () m ()') + einops.rearrange(depth_scores, 'b h n m -> b h () () n () () m')  # [1, Heads, H, W, D, H, W, D]
        content_scores /= 3
        content_scores = einops.rearrange(content_scores, 'b h i j k l m n -> b h (i j k) (l m n)')  # [1, Heads, #Content, #Content]

        # Attention score matrix
        # A (0) | B (0)
        # - - -
        # C | D
        scores = torch.zeros((cross_inst_content_scores.shape[0], self.heads, dim_q, dim_k), dtype=torch.float, device=device)
        if not self.no_bias_instructions:
            scores[:, :, dim_i:, :dim_i] = cross_inst_content_scores.unsqueeze(dim=-2).expand(-1, -1, dim_k - dim_i, -1)  # [B, Heads, #Content, I]. Matrix B
        if not self.no_bias_content:
            scores[:, :, dim_i:, dim_i:] = content_scores  # [B, Heads, #Content, #Content]. Matrix D

        # (Pre-)scale scores
        scores *= self.inv_temperature

        return scores

    def named_parameters_bias_content(self):

        params_ = [(name_, param_) for name_, param_ in self.named_parameters() if any([str_ in name_ for str_ in ['encoding_content', 'weights_content']])]

        return params_
        # return [self.encoding_content_h.named_parameters(), self.encoding_content_w, self.encoding_content_d,
        #         self.weights_content_h, self.weights_content_w, self.weights_content_d]

    def named_parameters_bias_instructions(self):

        params_ = [(name_, param_) for name_, param_ in self.named_parameters() if any([str_ in name_ for str_ in ['encoding_cross_inst_content', 'weights_cross_inst_content']])]

        return params_
        # return [self.encoding_cross_inst_content,
        #         self.weights_cross_inst_content]
