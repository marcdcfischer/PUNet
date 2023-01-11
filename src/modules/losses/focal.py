import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FocalLoss(nn.Module):
    def __init__(self,
                 out_channels: int,
                 loss_weight: float = 1.,
                 alpha_background: float = 0.1,
                 alpha_foreground: float = 0.1,
                 additive_alpha: Tuple[float, ...] = (0.0, 0.9),
                 gamma: float = 1.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.out_channels = out_channels
        self.alpha_background = alpha_background
        self.alpha_foreground = alpha_foreground
        self.additive_alpha = additive_alpha

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                label_indices_active: Optional[torch.Tensor] = None,
                tag: str = 'seg'):
        """
        :param predictions: logits [B, C, H, W, D]
        :param targets: int tensor [B, H, W, D]
        :param label_indices_active: [B, C]
        :return:
        """
        assert predictions[:, 0, ...].shape == targets.shape

        losses = dict()
        log_softmax = torch.clamp(F.log_softmax(predictions, dim=1), min=-1e3)
        log_prob = list()
        for idx_batch in range(targets.shape[0]):
            loss_weight_alpha = torch.tensor([self.alpha_background] + [self.alpha_foreground for _ in range(predictions.shape[1] - 1)], dtype=torch.float, device=predictions.device)
            loss_weight_alpha += torch.tensor(torch.tensor(self.additive_alpha)[torch.nonzero(label_indices_active[idx_batch, :], as_tuple=False).squeeze()] if label_indices_active is not None else self.additive_alpha, dtype=torch.float, device=predictions.device)
            assert predictions.shape[1] == loss_weight_alpha.shape[0]
            log_prob.append(F.nll_loss(input=log_softmax[idx_batch: idx_batch+1, ...],
                                       target=targets[idx_batch: idx_batch+1, ...],
                                       weight=loss_weight_alpha,
                                       reduction='none')[0, ...])
        log_prob = torch.stack(log_prob, dim=0)
        prob = torch.exp(-log_prob)

        losses[tag] = self.loss_weight * (torch.clamp(1. - prob, min=0.0) ** self.gamma * log_prob).mean()

        return losses
