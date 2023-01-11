import torch
import torch.nn as nn
from src.modules.blocks.query_encodings import LearnedNormedQuery, LearnedNormedInstruction
from typing import Tuple, Optional
import einops


class InstructionPool(nn.Module):
    def __init__(self,
                 instruction_pool_size: int,  # Atm expects an instruction pool size of 5
                 hidden_channels: int,
                 default_instructions: int,
                 tokens_per_instruction: int = 10,
                 separate_background: bool = True,
                 use_norm: bool = False,
                 elementwise_affine: bool = True):
        super().__init__()
        self.instruction_pool_size = instruction_pool_size
        self.hidden_channels = hidden_channels
        self.default_instructions = default_instructions
        self.separate_background = separate_background
        self.instruction_tokens = LearnedNormedInstruction(instruction_pool_size=instruction_pool_size,
                                                           tokens_per_instruction=tokens_per_instruction,
                                                           instruction_channels=hidden_channels,
                                                           use_norm=use_norm,
                                                           elementwise_affine=elementwise_affine)  # Pool with size of all possible combinations

    def forward(self, label_indices: Optional[torch.Tensor] = None, batch_size: Optional[int] = -1):
        """
        :param label_indices: None or [B, C]
        :return:
        """

        if label_indices is None:
            assert batch_size > 1  # If no label indices are given, a batch size is required.
            label_indices = torch.ones((batch_size, self.default_instructions), dtype=torch.long)  # [B, I_def]

        # Map label indices to corresponding instructions (atm combinations are hardcoded)
        instruction_tokens = self.instruction_tokens()  # [I, N, C]
        instructions = list()
        for idx_batch in range(label_indices.shape[0]):
            if self.separate_background:
                instruction_indices_true = torch.nonzero(label_indices[idx_batch, 1:], as_tuple=False).squeeze(dim=1) + 1  # All indices with designated background instruction ignored
            else:
                instruction_indices_true = torch.nonzero(label_indices[idx_batch, :], as_tuple=False).squeeze(dim=1)  # All indices including shared background
            selected_tokens = instruction_tokens[instruction_indices_true, ...]  # [I_active, N, C]
            instructions.append(einops.rearrange(selected_tokens, 'i n c -> (i n) c'))
            # Note: some instruction_tokens are not reachable (due to expected label ordering) and could be omitted.
        instructions = torch.stack(instructions, dim=0) if len(instructions) > 0 else None  # [B, (I N), C]

        return instructions
