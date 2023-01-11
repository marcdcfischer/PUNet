import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 widening_factor: int = 2):  # Perceiver used 4
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, in_channels * widening_factor)
        self.act_1 = nn.LeakyReLU()
        self.linear_2 = nn.Linear(in_channels * widening_factor, in_channels)

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.linear_2(x)
        return x
