import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from typing import Union, Dict, Optional, Tuple, List
from monai.networks.blocks import Convolution, ResidualUnit, UnetResBlock, UnetUpBlock
from src.modules.blocks.sia_res_block_deep import DeepSIAResBlock
from src.modules.blocks.sia_up_block_deep import DeepSIAUpBlock
from src.modules.blocks.instruction_pool import InstructionPool
from src.modules.blocks.similarity_aggregation import similarity_aggregation
import collections
from einops import rearrange
import math
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import warnings
from itertools import chain


class DeepSIAUNetr(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf
        self.bias_vit = self.conf.bias_vit

        # Segmentation instructions (atm one token per class)
        assert self.conf.instruction_pool_size >= self.conf.out_channels
        self.separate_background = self.conf.separate_background and self.conf.label_indices_max_active + 1 < 2
        if self.separate_background:
            print('Using separate background tokens for each foreground class.')
        else:
            print('Using shared background tokens for all foreground classes.')
        self.tokens_per_instruction_combined = 2 * self.conf.tokens_per_instruction if self.separate_background else self.conf.tokens_per_instruction

        # Make sure variants are configured (somewhat) properly
        assert self.conf.instruction_channels == self.conf.hidden_channels[0]
        if 'prompting' in self.conf.adaptation_variant.lower():
            assert self.conf.fixed_output is False
        else:
            assert self.conf.fixed_output is True

        # Make sure dimensions are divisible at least by 2 or are equal to 1
        for patch_size_ in ([self.conf.patch_size_teacher, *self.conf.patch_size_students]):
            assert all([x_ % 2 == 0 or x_ == 1 for x_ in patch_size_])

        # Architecture
        # CNN Encoder (with only one downsampling)
        self.depth_kernel_size_cnn = [3 if self.conf.patch_size_students[0][2] >= 3. else 1 for _ in range(self.conf.depth_cnn_encoder)]
        # self.depth_stride_cnn = [2 if self.conf.patch_size_students[0][2] > 1. else 1 for _ in range(self.conf.depth_cnn_encoder)]
        self.encoding_res_blocks = nn.ModuleList()
        for idx_block in range(self.conf.depth_cnn_encoder):
            self.encoding_res_blocks.append(UnetResBlock(
                spatial_dims=3,
                in_channels=self.conf.hidden_channels[0] if idx_block > 0 else self.conf.in_channels,
                out_channels=self.conf.hidden_channels[0],
                kernel_size=(3, 3, self.depth_kernel_size_cnn[idx_block]),
                stride=(1, 1, 1) if idx_block + 1 < self.conf.depth_cnn_encoder else (2, 2, 1),  # Downsampling only for last CNN layer and only intra-plane
                norm_name="batch",
                dropout=0.0,
            ))

        # (Optional) ViT position encoding
        self.max_patches = [math.ceil(x_ / 2) for x_ in self.conf.patch_size_teacher]  # Teacher is expected to have the highest size.
        if self.bias_vit:
            self.vit_pos_bias_content = nn.Parameter(nn.init.xavier_uniform_(torch.empty((1, math.prod(self.max_patches), self.conf.hidden_channels[0])),
                                                                             gain=nn.init.calculate_gain('linear')), requires_grad=True)  # monai used truncated normal
            self.vit_pos_bias_instructions = nn.Parameter(nn.init.xavier_uniform_(torch.empty((1, 1, self.conf.hidden_channels[0])),
                                                                                  gain=nn.init.calculate_gain('linear')), requires_grad=True)

        # UNet Encoder (with SIA blocks)
        self.depth_kernel_size_unet = [3 if self.conf.patch_size_students[0][2] / 2 ** idx_block >= 3. else 1 for idx_block in range(self.conf.depth_sia_encoder)]
        self.depth_stride_unet = [1] + [2 if self.conf.patch_size_students[0][2] / 2 ** idx_block > 1. else 1 for idx_block in range(1, self.conf.depth_sia_encoder)]
        # self.depth_scale_factor_unet = [2. if self.conf.patch_size_students[0][2] / 2 ** idx_block > 1. else 1 for idx_block in range(self.conf.depth_sia_encoder)]
        self.sia_res_blocks = nn.ModuleList()
        self.sia_res_instruction_blocks_v2 = nn.ModuleList()
        for idx_block in range(self.conf.depth_sia_encoder):
            print(f'Encoder - Using kernel size {(3, 3, self.depth_kernel_size_unet[idx_block])}, strides {(2, 2, self.depth_stride_unet[idx_block]) if idx_block > 0 else (1, 1, 1)}.')
            self.sia_res_blocks.append(DeepSIAResBlock(
                channels_in=self.conf.hidden_channels[idx_block - 1] if idx_block > 0 else self.conf.hidden_channels[idx_block] if self.conf.depth_cnn_encoder > 0 else self.conf.in_channels,
                channels_out=self.conf.hidden_channels[idx_block],
                instruction_pool_size=self.conf.instruction_pool_size,  # kind of an upperlimit.
                tokens_per_instruction=self.tokens_per_instruction_combined,
                separate_background=self.conf.separate_background,
                kernel_size=(3, 3, self.depth_kernel_size_unet[idx_block]),
                strides=(2, 2, self.depth_stride_unet[idx_block]) if idx_block > 0 else (1, 1, 1),
                heads=self.conf.attention_heads,
                window_size=self.conf.attn_window_size,
                unique_instruction_bias=self.conf.unique_instruction_bias,
                unique_token_bias=self.conf.unique_token_bias,
                no_bias_instructions=self.conf.no_bias_instructions,
                no_bias_content=self.conf.no_bias_content,
                adapter=self.conf.adaptation_variant.lower() == 'adapter',
            ))  # TODO: Make same check as e.g. for depth_kernel_size for passed window_size entries

            if ((idx_block == 0 and self.conf.prompting_variant.lower() in ['start', 'encoder', 'full'])\
                    or (idx_block > 0 and self.conf.prompting_variant.lower() in ['encoder', 'full']))\
                    and not self.conf.fixed_output:
                for idx_sub_block_ in range(2):  # w and sw
                    self.sia_res_instruction_blocks_v2.append(InstructionPool(
                        instruction_pool_size=self.conf.instruction_pool_size,
                        hidden_channels=self.conf.hidden_channels[idx_block],
                        default_instructions=self.conf.out_channels,
                        tokens_per_instruction=self.tokens_per_instruction_combined,
                        separate_background=self.conf.separate_background,
                        use_norm=self.conf.instructions_use_norm,
                        elementwise_affine=self.conf.instructions_elementwise_affine
                    ))
            else:
                self.sia_res_instruction_blocks_v2.extend([nn.Module(), nn.Module()])

        # UNet Decoder (with SIA blocks)
        self.sia_up_blocks = nn.ModuleList()
        self.sia_up_instruction_blocks_v2 = nn.ModuleList()
        for idx_block in range(self.conf.depth_sia_decoder):
            print(f'Decoder - Using kernel size {(3, 3, list(reversed(self.depth_kernel_size_unet))[idx_block])}, strides {(2, 2, list(reversed(self.depth_stride_unet))[idx_block])}.')
            self.sia_up_blocks.append(DeepSIAUpBlock(
                channels_in=list(reversed(self.conf.hidden_channels[:len(self.sia_res_blocks)]))[idx_block],
                channels_out=list(reversed(self.conf.hidden_channels[:len(self.sia_res_blocks)]))[idx_block + 1],
                instruction_pool_size=self.conf.instruction_pool_size,
                tokens_per_instruction=self.tokens_per_instruction_combined,
                separate_background=self.conf.separate_background,
                kernel_size=(3, 3, list(reversed(self.depth_kernel_size_unet))[idx_block]),
                strides=(2, 2, list(reversed(self.depth_stride_unet))[idx_block]),
                heads=self.conf.attention_heads,
                window_size=self.conf.attn_window_size,
                unique_instruction_bias=self.conf.unique_instruction_bias,
                unique_token_bias=self.conf.unique_token_bias,
                no_bias_instructions=self.conf.no_bias_instructions,
                no_bias_content=self.conf.no_bias_content,
                adapter=self.conf.adaptation_variant.lower() == 'adapter',
            ))

            if self.conf.prompting_variant.lower() in ['decoder', 'full'] and not self.conf.fixed_output:
                for idx_sub_block_ in range(2):  # w and sw
                    self.sia_up_instruction_blocks_v2.append(InstructionPool(
                        instruction_pool_size=self.conf.instruction_pool_size,
                        hidden_channels=list(reversed(self.conf.hidden_channels[:len(self.sia_res_blocks)]))[idx_block + 1],
                        default_instructions=self.conf.out_channels,
                        tokens_per_instruction=self.tokens_per_instruction_combined,
                        separate_background=self.conf.separate_background,
                        use_norm=self.conf.instructions_use_norm,
                        elementwise_affine=self.conf.instructions_elementwise_affine
                    ))
            else:
                self.sia_up_instruction_blocks_v2.extend([nn.Module(), nn.Module()])

        # Final upsampling
        if self.conf.depth_cnn_encoder > 0:
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=False)

        # Last pool is always active (otherwise no sim comparison is possible)
        # I.e. this pool exists regardless of start, end, encoder, decoder, full variants
        if not self.conf.fixed_output:
            self.instruction_pool = InstructionPool(instruction_pool_size=self.conf.instruction_pool_size,
                                                    hidden_channels=self.conf.instruction_channels,
                                                    default_instructions=self.conf.out_channels,
                                                    tokens_per_instruction=self.tokens_per_instruction_combined,
                                                    separate_background=self.separate_background,
                                                    use_norm=self.conf.instructions_use_norm,
                                                    elementwise_affine=self.conf.instructions_elementwise_affine)
        # Fixed output (ablation)
        else:
            self.norm_fixed = get_norm_layer(name='batch',
                                             spatial_dims=3,
                                             channels=self.conf.hidden_channels[0])
            self.conv_fixed = nn.Conv3d(in_channels=self.conf.hidden_channels[0],
                                        out_channels=self.conf.label_indices_max_active + 1 if self.conf.label_indices_max_active > 0 else self.conf.out_channels,
                                        kernel_size=(1, 1, 1))

        # Mean rep initialization
        if self.conf.mean_initialization and self.conf.downstream:
            self.set_downstream_instruction_parameters(label_indices_base=self.conf.label_indices_base,
                                                       label_indices_downstream=self.conf.label_indices_downstream_active)

    def get_named_cnn_encoder_parameters(self):

        params_ = list(self.encoding_res_blocks.named_parameters())

        return params_

    def get_named_fixed_parameters(self):

        params_ = []
        if self.conf.fixed_output:
            params_ = list(chain(*[self.norm_fixed.named_parameters(), self.conv_fixed.named_parameters()]))

        return params_

    def get_named_encoder_parameters(self):

        params_ = list(chain(*[x_.named_parameters_body() for x_ in self.sia_res_blocks]))

        return params_

    def get_named_decoder_parameters(self):

        params_ = list(chain(*[x_.named_parameters_body() for x_ in self.sia_up_blocks]))

        return params_

    def get_named_body_parameters(self):

        params_ = list(chain(*[self.get_named_encoder_parameters(), self.get_named_decoder_parameters(), self.get_named_cnn_encoder_parameters(), self.get_named_fixed_parameters()]))

        return params_

    def get_named_instruction_bias_parameters(self):

        sia_res_blocks_instruction_bias_params = list(chain(*[x_.named_parameters_bias_instructions() for x_ in self.sia_res_blocks]))
        sia_up_blocks_instruction_bias_params = list(chain(*[x_.named_parameters_bias_instructions() for x_ in self.sia_up_blocks]))
        params_ = [*sia_res_blocks_instruction_bias_params, *sia_up_blocks_instruction_bias_params]

        return params_

    def get_named_instruction_pool_parameters(self):

        params_ = list(chain(*[self.sia_res_instruction_blocks_v2.named_parameters(), self.sia_up_instruction_blocks_v2.named_parameters()]))
        if not self.conf.fixed_output:
            params_ += list(self.instruction_pool.named_parameters())

        return params_

    def get_named_instruction_parameters(self):

        params_ = list(chain(*[self.get_named_instruction_pool_parameters(), self.get_named_instruction_bias_parameters()]))

        return params_

    def get_named_adapter_parameters(self):

        params_ = list(chain(*[x_.named_parameters_adapter() for x_ in self.sia_res_blocks]))
        params_ += list(chain(*[x_.named_parameters_adapter() for x_ in self.sia_up_blocks]))

        return params_

    def set_requires_gradient(self,
                              grad_instructions: torch.BoolTensor,
                              grad_instructions_norm: bool = True,
                              grad_instructions_scores: bool = True,
                              grad_body: bool = True):

        # Disable everything (as default)
        debug = True
        for (name_, param_student_) in self.named_parameters():
            param_student_.requires_grad = False

            if debug:
                # Make sure every parameter is included in any of the submodules
                if not any([param_student_.data_ptr() == x_[1].data_ptr() for x_ in [*self.get_named_body_parameters(), *self.get_named_instruction_parameters(), *self.get_named_adapter_parameters()]]):
                    raise ValueError(f'Missing parameter {name_}')

        # Enable or disable gradients for bulk of interpreter (student) - dependent on selective freezing
        for (name_, param_student_) in self.get_named_body_parameters():
            param_student_.requires_grad = grad_body

        # Last layer
        if self.conf.adaptation_variant in ['fixed', 'decoder', 'bias', 'adapter']:
            for (name_, param_student_) in self.get_named_fixed_parameters():
                param_student_.requires_grad = True

        # Adapter layers
        if self.conf.adaptation_variant in ['adapter']:
            for (name_, param_student_) in self.get_named_adapter_parameters():
                param_student_.requires_grad = True

        # Decoder
        if self.conf.adaptation_variant in ['decoder']:
            for (name_, param_student_) in self.get_named_decoder_parameters():
                param_student_.requires_grad = True

        # Enable or disable gradients for bias params in bulk of interpreter
        if self.conf.adaptation_variant in ['bias', 'bias_prompting']:
            for (name_, param_student_) in self.get_named_body_parameters():
                if 'bias' in name_ or 'norm' in name_:  # Includes bias + scale parameters of norms
                    param_student_.requires_grad = True

        # Enable or disable gradients for all instruction parameters
        if self.conf.adaptation_variant in ['prompting', 'bias_prompting']:
            # Only active instruction bias scores are adjusted.
            for (name_, param_student_) in self.get_named_instruction_bias_parameters():
                if 'encoding_cross_inst_content' in name_:
                    if int(name_[-1]) < grad_instructions.shape[0]:
                        param_student_.requires_grad = grad_instructions[int(name_[-1])].item() & grad_instructions_scores
                    else:
                        param_student_.requires_grad = False  # Excess instruction remain unused and therefore default False
                else:
                    param_student_.requires_grad = grad_instructions_scores

            # Fine-grained instruction pool adjustments
            for (name_, param_student_) in self.get_named_instruction_pool_parameters():
                # Set token parameter
                if 'instructions_norm' in name_:  # norm weights and bias
                    param_student_.requires_grad = grad_instructions_norm
                else:  # instructions.0+
                    if int(name_[-1]) < grad_instructions.shape[0]:
                        param_student_.requires_grad = grad_instructions[int(name_[-1])].item()
                    else:
                        param_student_.requires_grad = False  # Excess instruction remain unused and therefore default False

        # Report frozen / nonfrozen
        def _report_trainable(key_, named_params):
            print(f"Trainable are {sum([p_[1].numel() for p_ in named_params if p_[1].requires_grad])}/{sum([p_[1].numel() for p_ in named_params])} {key_} parameters.")

        print(f"Trainable parameters for adaptation variant {self.conf.adaptation_variant}.")
        _report_trainable('cnn encoder', self.get_named_cnn_encoder_parameters())
        _report_trainable('encoder', self.get_named_encoder_parameters())
        _report_trainable('decoder', self.get_named_decoder_parameters())
        _report_trainable('fixed layer', self.get_named_fixed_parameters())
        _report_trainable('body', self.get_named_body_parameters())
        _report_trainable('instruction bias', self.get_named_instruction_bias_parameters())  # Note: amount of truly active bias parameters may be less (for position ablations).
        _report_trainable('instruction pool', self.get_named_instruction_pool_parameters())
        _report_trainable('instruction', self.get_named_instruction_parameters())
        _report_trainable('adapter', self.get_named_adapter_parameters())
        _report_trainable('all', list(self.named_parameters()))

    def set_downstream_instruction_parameters(self, label_indices_base: List[int], label_indices_downstream: List[int]):

        if len(label_indices_base) > 0:
            print('Performing initialization of instructions based on mean representation.')
            mean_rep = torch.mean(torch.stack(list(self.instruction_pool.instruction_tokens.instructions), dim=0)[label_indices_base, ...], dim=0)  # Mean rep of existing foreground categories
            for idx_instruction, tokens_ in enumerate(self.instruction_pool.instruction_tokens.instructions):
                if idx_instruction in label_indices_downstream:
                    if idx_instruction in label_indices_base:
                        warnings.warn(f'Initializing downstream instruction {idx_instruction} with mean rep despite it being present in label_indices_base. (Ignore if intended.)')
                    tokens_.data.copy_(mean_rep)
        else:
            print('Initialization of instructions remains random since no base labels are available.')

    def forward(self,
                x: torch.Tensor,
                label_indices: Optional[torch.Tensor] = None,
                pseudo_indices_subject: Optional[torch.Tensor] = None,
                pseudo_indices_label: Optional[torch.Tensor] = None,
                mode_label: str = 'pseudo',
                mode_loss: str = 'both'):  # pseudo or label
        dict_out = collections.defaultdict(dict)

        # Fetch instructions
        # Note: mode does not have an effect atm. Pseudo does not exist for deep (since it would be too large)

        # CNN Encoding (with only one downsampling step)
        for idx_block in range(self.conf.depth_cnn_encoder):
            x = self.encoding_res_blocks[idx_block](x)

        if self.bias_vit:
            vit_pos_bias_ = rearrange(self.vit_pos_bias_content, 'b (h w d) c -> b c h w d', h=self.max_patches[0], w=self.max_patches[1], d=self.max_patches[2])
            x = x + vit_pos_bias_[...,
                                  math.floor((vit_pos_bias_.shape[2] - x.shape[2]) / 2): vit_pos_bias_.shape[2] - math.ceil((vit_pos_bias_.shape[2] - x.shape[2]) / 2),
                                  math.floor((vit_pos_bias_.shape[3] - x.shape[3]) / 2): vit_pos_bias_.shape[3] - math.ceil((vit_pos_bias_.shape[3] - x.shape[3]) / 2),
                                  math.floor((vit_pos_bias_.shape[4] - x.shape[4]) / 2): vit_pos_bias_.shape[4] - math.ceil((vit_pos_bias_.shape[4] - x.shape[4]) / 2)]
            # x_instructions = x_instructions + self.vit_pos_bias_instructions

        # Transformer processing
        # Encoder
        dict_out['dense']['skips'] = list()
        dict_out['instructions']['skips'] = list()
        for idx_block in range(self.conf.depth_sia_encoder):
            if mode_loss == 'self'\
                    or (self.conf.noninstructed_attention and not self.conf.downstream)\
                    or (self.conf.noninstructed_attention_downstream and self.conf.downstream)\
                    or not isinstance(self.sia_res_instruction_blocks_v2[2 * idx_block], InstructionPool):  # no instructions for mode_loss == 'self' (only)
                x = self.sia_res_blocks[idx_block](x=x,
                                                   x_instructions=None,
                                                   label_indices=None)
            else:
                x_instructions = [self.sia_res_instruction_blocks_v2[2 * idx_block](label_indices, batch_size=x.shape[0]),
                                  self.sia_res_instruction_blocks_v2[2 * idx_block + 1](label_indices, batch_size=x.shape[0])]
                x = self.sia_res_blocks[idx_block](x=x,
                                                   x_instructions=x_instructions,
                                                   label_indices=label_indices)
            if idx_block + 1 < self.conf.depth_sia_encoder:
                dict_out['dense']['skips'].append(x)

        # Decoder
        for idx_block in range(len(self.sia_up_blocks)):
            if mode_loss == 'self'\
                    or (self.conf.noninstructed_attention and not self.conf.downstream)\
                    or (self.conf.noninstructed_attention_downstream and self.conf.downstream)\
                    or not isinstance(self.sia_up_instruction_blocks_v2[2 * idx_block], InstructionPool):
                x = self.sia_up_blocks[idx_block](x=x,
                                                  x_skips=list(reversed(dict_out['dense']['skips']))[idx_block],
                                                  x_instructions=None,
                                                  label_indices=None)
            else:
                x_instructions = [self.sia_up_instruction_blocks_v2[2 * idx_block](label_indices, batch_size=x.shape[0]),
                                  self.sia_up_instruction_blocks_v2[2 * idx_block + 1](label_indices, batch_size=x.shape[0])]
                x = self.sia_up_blocks[idx_block](x=x,
                                                  x_skips=list(reversed(dict_out['dense']['skips']))[idx_block],
                                                  x_instructions=x_instructions,
                                                  label_indices=label_indices)

        dict_out['patched']['embedded_latents'] = x

        # Segmentation recombination
        if not self.conf.fixed_output:
            x_instructions_final = self.instruction_pool(label_indices, batch_size=x.shape[0])
            dict_out['instructions']['segmentation_latents'] = x_instructions_final

            h_, w_, d_ = dict_out['patched']['embedded_latents'].shape[-3:]
            x_sim_latents = einops.rearrange(dict_out['patched']['embedded_latents'], 'b c h w d -> b (h w d) c')
            x_sim_instructions = einops.rearrange(dict_out['instructions']['segmentation_latents'], 'b (i n) c -> b i n c', n=self.conf.tokens_per_instruction)  # [B, I, N, C]. Should add up to the same form regardless of self.separate_background (for binary case).
            x_sim = similarity_aggregation(latents=x_sim_latents,
                                           instructions=x_sim_instructions,
                                           mean_aggregation=self.conf.mean_aggregation,
                                           top_k_selection=self.conf.top_k_selection,
                                           soft_selection_sigma=self.conf.soft_selection_sigma)
            x_sim = einops.rearrange(x_sim, 'b i (h w d) -> b i h w d', h=h_, w=w_, d=d_)
        else:
            dict_out['instructions']['segmentation_latents'] = None

            # assert self.conf.architecture == 'wip_simple'  # Should only be used in conjunction with simple (multiclass) case.
            x_sim = self.conv_fixed(F.leaky_relu(self.norm_fixed(dict_out['patched']['embedded_latents']), inplace=True))

        dict_out['dense']['embedded_latents'] = self.up(x_sim) if self.conf.depth_cnn_encoder > 0 else x_sim

        return dict_out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--depth_cnn_encoder', default=1, type=int)
        parser.add_argument('--depth_sia_encoder', default=5, type=int)
        parser.add_argument('--depth_sia_decoder', default=4, type=int)
        parser.add_argument('--instruction_pool_size', default=10, type=int)
        parser.add_argument('--instruction_pool_size_pseudo_subjects', default=100, type=int)
        parser.add_argument('--instruction_pool_size_pseudo_labels', default=51, type=int)
        parser.add_argument('--instruction_channels', default=32, type=int)
        parser.add_argument('--tokens_per_instruction', default=16, type=int)
        # parser.add_argument('--tokens_per_background', default=20, type=int)
        parser.add_argument('--attention_heads', default=8, type=int)
        parser.add_argument('--attn_window_size', default=[8, 8, 1], nargs=3, type=int)
        parser.add_argument('--hidden_channels', default=[32, 64, 128, 256, 384], nargs='*', type=int)

        # Instruction initialization / aggregation
        parser.add_argument('--noninstructed_attention', action='store_true')  # Attention layers are not instructed
        parser.add_argument('--noninstructed_attention_downstream', action='store_true')
        parser.add_argument('--top_k_selection', action='store_true')  # True: Aggregate via softmax re-weighting, False: Mean over topk (atm 3)
        parser.add_argument('--soft_selection_sigma', default=0.1, type=float)  # Temperature for softmax re-weighting
        parser.add_argument('--mean_aggregation', action='store_true')  # Aggregate instructions without any sophisticated selection
        parser.add_argument('--mean_initialization', default=False, type=bool)  # Use mean representation of learned (base) categories for initialization of new (unseen) downstream category
        parser.add_argument('--fixed_output', action='store_true')  # Fixed linear output layer instead of cosine similarity matching with instructions.
        parser.add_argument('--instructions_use_norm', default=False, type=bool)  # Norm all instructions in a pool by a common norm.
        parser.add_argument('--instructions_elementwise_affine', default=True, type=bool)  # Enable / disable learning of extra norm params for instructions.
        parser.add_argument('--prompting_variant', default='full', type=str, choices=['start', 'end', 'encoder', 'decoder', 'full'])
        parser.add_argument('--adaptation_variant', default='prompting', type=str, choices=['prompting', 'fixed', 'decoder', 'bias', 'adapter', 'bias_prompting'])

        # Attention bias scheme
        parser.add_argument('--unique_instruction_bias', default=True, type=bool)  # If True each Instruction has a unique bias score.
        parser.add_argument('--unique_token_bias', default=True, type=bool)  # If True each Token (across all instructions) has a unique bias score. IF False all bias scores are the same regardless of the token.
        parser.add_argument('--no_bias_instructions', action='store_true')
        parser.add_argument('--no_bias_content', action='store_true')
        parser.add_argument('--bias_vit', action='store_true')

        return parser
