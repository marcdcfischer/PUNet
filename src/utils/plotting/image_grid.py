import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import itertools
import os
import math
from typing import Optional, List
import torch.nn.functional as F


def plot_grid_middle(x,
                     targets: Optional[torch.Tensor] = None,
                     preds: Optional[torch.Tensor] = None,
                     scribbles: Optional[torch.Tensor] = None,
                     indices_elements: List[int] = (0, 4, 8),
                     prefix='val', dpi=200, axes_size=8, path_plots=None):
    path_plots = os.path.join(path_plots, 'plots') if path_plots is not None else os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'logs', 'lightning', 'plots')

    os.makedirs(path_plots, exist_ok=True)
    with torch.no_grad():
        x = x[..., x.shape[4]//2]
        targets = targets[..., targets.shape[4]//2]
        preds = preds[..., preds.shape[4]//2]
        scribbles = scribbles[..., scribbles.shape[3]//2] if scribbles is not None else None
        channels = targets.shape[1]
        # zoom elements
        zoom_input_x, zoom_input_y = preds.shape[2] / x.shape[2], x.shape[3] / x.shape[3]
        zoom_target_x, zoom_target_y = preds.shape[2] / targets.shape[2], preds.shape[3] / targets.shape[3]
        if zoom_input_x < 1. or zoom_target_y < 1.:
            x_zoomed = F.interpolate(x, size=preds.shape[2:])
        else:
            x_zoomed = x
        if zoom_target_x < 1. or zoom_target_y < 1.:
            targets_zoomed = F.one_hot(F.interpolate(targets.argmax(dim=1, keepdim=True), size=preds.shape[2:], mode='nearest').squeeze(1), num_classes=channels).permute(0, 3, 1, 2)
            scribbles_zoomed = F.interpolate(scribbles.unsqueeze(dim=1), size=preds.shape[2:], mode='nearest').squeeze(dim=1) if scribbles is not None else None
        else:
            targets_zoomed = targets
            scribbles_zoomed = scribbles

        png_paths = list()
        for idx_img in [min(x.shape[0] - 1, n_) for n_ in indices_elements]:
            targets_slices = [targets_zoomed[idx_img:idx_img + 1, idx_:idx_ + 1, ...].float() for idx_ in range(channels)]
            preds_slices = [preds[idx_img:idx_img + 1, idx_:idx_ + 1, ...].float() for idx_ in range(channels)]
            paired_slices = list(itertools.chain(*list(zip(targets_slices, preds_slices))))
            grid = make_grid(torch.cat(
                [((x_zoomed[idx_img:idx_img + 1, ...]) - torch.min(x_zoomed[idx_img:idx_img + 1, ...])) / (torch.max(x_zoomed[idx_img:idx_img + 1, ...] - torch.min(x_zoomed[idx_img:idx_img + 1, ...]))),
                 scribbles_zoomed[idx_img:idx_img + 1, ...].unsqueeze(dim=1).float() / channels if scribbles is not None else torch.zeros_like(x_zoomed[idx_img:idx_img + 1, ...]),
                 torch.argmax(targets_zoomed[idx_img:idx_img + 1, ...], dim=1, keepdim=True).float() / channels,
                 torch.argmax(preds[idx_img:idx_img + 1, ...], dim=1, keepdim=True).float() / channels,
                 *paired_slices
                 ], dim=1).permute(1, 0, 2, 3), padding=5).numpy().transpose(1, 2, 0)

            fig, ax = plt.subplots(figsize=(axes_size * int(math.ceil(channels / 8.)), axes_size * 8))
            ax.axis('off')
            fig.tight_layout()
            ax.imshow(grid)
            png_paths.append(f'{path_plots}/{prefix}_{str(idx_img).zfill(2)}_grid.png')
            plt.savefig(png_paths[-1], bbox_inches='tight', dpi=dpi)
            plt.close(fig)

    return png_paths
