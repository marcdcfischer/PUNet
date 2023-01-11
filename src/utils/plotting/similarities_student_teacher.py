import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Optional, Dict
from scipy import ndimage
import torch
import pathlib as plb


def visualize_similarities_student_teacher(plots: dict,
                                           x_student: torch.Tensor,
                                           x_teacher: torch.Tensor,
                                           y_teacher: torch.Tensor,
                                           n_elements: int = 6,
                                           dpi: int = 200,
                                           prefix: str = 'val',
                                           path_plots: Optional[str] = None):
    """Visualizes softmaxed similarities"""

    # detach items
    x_student = x_student.detach().cpu()
    x_teacher = x_teacher.detach().cpu()

    png_paths = list()
    for key_, value_ in plots.items():
        if any([s_ in key_ for s_ in ['sim_student_teacher']]):
            n_elements_ = min([n_elements, value_.shape[0], value_.shape[4]])
            png_paths.append(plot_sim_student_teacher(sim_data=np.array(value_.detach().cpu().float()),
                                                      x_student=np.array(x_student),
                                                      x_teacher=np.array(x_teacher),
                                                      y_teacher=np.array(y_teacher.cpu()),
                                                      n_elements=n_elements_,
                                                      path_plots=path_plots,
                                                      prefix=prefix,
                                                      dpi=dpi,
                                                      tag=key_))

    return png_paths


def plot_sim_student_teacher(sim_data: np.ndarray,
                             x_student: np.ndarray,
                             x_teacher: np.ndarray,
                             y_teacher: np.ndarray,
                             n_elements: int = 6,
                             axes_size: float = 2.,
                             path_plots: Optional[str] = None,
                             prefix: str = 'val',
                             dpi: int = 200,
                             tag: str = ''):
    path_plots = os.path.join(path_plots, 'plots') if path_plots is not None else str(plb.Path(__file__).resolve().parent.parent.parent / 'logs' / 'lightning' / 'plots')
    os.makedirs(path_plots, exist_ok=True)

    # normalize data
    x_student = np.stack([(x_student[idx_] - np.amin(x_student[idx_])) / (np.amax(x_student[idx_]) - np.amin(x_student[idx_]) + 1e-12) for idx_ in range(x_student.shape[0])], axis=0)  # per img
    x_teacher = np.stack([(x_teacher[idx_] - np.amin(x_teacher[idx_])) / (np.amax(x_teacher[idx_]) - np.amin(x_teacher[idx_]) + 1e-12) for idx_ in range(x_teacher.shape[0])], axis=0)  # per img
    sim_data = (sim_data - np.amin(sim_data)) / (np.amax(sim_data) - np.amin(sim_data) + 1e-12)  # overall

    # Select elements
    x_student = x_student[:n_elements, 0, ...]  # [N, H, W, D]. Only first channel
    x_teacher = x_teacher[:n_elements, 0, ...]
    y_teacher = y_teacher[:n_elements, ...]
    x_shape_diff = [x_t - x_s for x_t, x_s in zip(x_teacher.shape[1:], x_student.shape[1:])]
    x_student_padded = np.pad(x_student,
                              ((0, 0),
                               (x_shape_diff[0] // 2, x_shape_diff[0] // 2 + x_shape_diff[0] % 2),
                               (x_shape_diff[1] // 2, x_shape_diff[1] // 2 + x_shape_diff[1] % 2),
                               (x_shape_diff[2] // 2, x_shape_diff[2] // 2 + x_shape_diff[2] % 2)))
    sim_data_shape = sim_data.shape
    sim_data_selected = sim_data[:n_elements,
                        sim_data_shape[1] // 2,
                        sim_data_shape[2] // 2,
                        sim_data_shape[3] // 2,
                        :, :, :].reshape((n_elements, sim_data_shape[-3], sim_data_shape[-2], sim_data_shape[-1]))  # [N, H, W, D]
    zoom_x, zoom_y, zoom_z = x_teacher.shape[1] / sim_data_selected.shape[1], x_teacher.shape[2] / sim_data_selected.shape[2], x_teacher.shape[3] / sim_data_selected.shape[3]
    sim_data_zoomed = np.stack([ndimage.zoom(sim_data_selected[idx_], zoom=(zoom_x, zoom_y, zoom_z), order=0) for idx_ in range(sim_data_selected.shape[0])], axis=0)

    # Simple cross section view
    x_student_padded = x_student_padded[..., x_student_padded.shape[-1] // 2]
    x_teacher = x_teacher[..., x_teacher.shape[-1] // 2]
    y_teacher = y_teacher[..., y_teacher.shape[-1] // 2]
    sim_data_zoomed = sim_data_zoomed[..., sim_data_zoomed.shape[-1] // 2]

    # Image grid
    x_images = torch.Tensor(np.concatenate([np.rot90(x_student_padded, 1, axes=(-2, -1)), np.rot90(x_teacher, axes=(-2, -1))], axis=0)).unsqueeze(1)
    x_sim_1 = torch.Tensor(np.concatenate([np.zeros_like(np.rot90(sim_data_zoomed, axes=(-2, -1))), np.rot90(sim_data_zoomed, axes=(-2, -1))], axis=0)).unsqueeze(1)
    x_sim_2 = 1 - torch.Tensor(np.concatenate([np.ones_like(np.rot90(sim_data_zoomed, axes=(-2, -1))), np.rot90(sim_data_zoomed, axes=(-2, -1))], axis=0)).unsqueeze(1)
    y_overlay = torch.Tensor(np.concatenate([np.zeros_like(np.rot90(x_student_padded, 1, axes=(-2, -1))), np.rot90(y_teacher, axes=(-2, -1))], axis=0)).unsqueeze(1)
    grid_images = make_grid(x_images, nrow=n_elements, normalize=False).numpy().transpose(1, 2, 0)
    grid_sim_1 = make_grid(x_sim_1, nrow=n_elements, normalize=False).numpy().transpose(1, 2, 0)
    grid_sim_2 = make_grid(x_sim_2, nrow=n_elements, normalize=False).numpy().transpose(1, 2, 0)
    grid_overlay = make_grid(y_overlay, nrow=n_elements, normalize=False).numpy().transpose(1, 2, 0)
    pos_y = x_teacher.shape[1] // 2
    pos_x = x_teacher.shape[2] // 2
    grid_pos_x = [pos_x + 2 + idx_x % n_elements * (2 + 2 * pos_x) for _ in range(2) for idx_x in range(n_elements)]
    grid_pos_y = [pos_y + 2 + idx_y // 1 * (2 + 2 * pos_y) for idx_y in range(2) for _ in range(n_elements)]

    n_rows, n_cols = 2, n_elements
    cmap_ = plt.cm.get_cmap('jet')
    cmap_.set_under(color='k', alpha=0)
    cmap_2 = plt.cm.get_cmap('cool')
    cmap_2.set_under(color='k', alpha=0)
    cmap_3 = plt.cm.get_cmap('Blues')
    cmap_3.set_under(color='k', alpha=0)
    fig, ax = plt.subplots(figsize=(axes_size * n_cols, axes_size * n_rows))
    ax.axis('off')
    ax.imshow(grid_images[..., 0], cmap='gray')
    content_ = grid_sim_1[..., 0]
    vmax_ = content_.max()
    vmin_ = (content_.max() - content_.min()) / 2
    im = ax.imshow(content_, cmap=cmap_, alpha=0.4 * (np.maximum(content_, vmin_) - vmin_) / (vmax_ - vmin_), vmin=vmin_, vmax=vmax_, interpolation='nearest')
    content_ = grid_sim_2[..., 0]
    vmax_ = content_.max()
    vmin_ = (content_.max() - content_.min()) / 2
    im = ax.imshow(content_, cmap=cmap_2, alpha=0.6 * (np.maximum(content_, vmin_) - vmin_) / (vmax_ - vmin_), vmin=0.1, vmax=vmax_, interpolation='nearest')
    content_ = grid_overlay[..., 0]
    vmax_ = max(content_.max(), 1)
    vmin_ = content_.min() + 0.01
    im = ax.imshow(content_, cmap=cmap_3, alpha=0.75, vmin=vmin_, vmax=vmax_, interpolation='nearest')
    # fig.colorbar(im)
    # ax.scatter(grid_pos_x, grid_pos_y, s=2, c='magenta')
    png_path = f'{path_plots}/{prefix}_{tag}.png'
    fig.tight_layout()
    plt.savefig(png_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    return png_path
