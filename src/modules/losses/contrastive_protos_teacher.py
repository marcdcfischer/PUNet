import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, Tuple, List, Optional
import math
import einops
import numpy as np


class ContrastiveProtosTeacherLoss(nn.Module):
    def __init__(self,
                 reduction_factor: float = 4.,  # Grid sampling to make loss calc feasible
                 reduction_factor_protos: float = 16.,  # Undersampling factor for prototype seeds
                 loss_weight: float = 1.0,
                 k_means_iterations: int = 3,
                 use_weighting_protos: bool = True,
                 use_weighting_teacher: bool = False,
                 fwhm_student_teacher: float = 128.,
                 fwhm_teacher_protos: float = 128.):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.reduction_factor_protos = reduction_factor_protos
        self.loss_weight = loss_weight
        self.k_means_iterations = k_means_iterations
        self.use_weighting_protos = use_weighting_protos
        self.use_weighting_teacher = use_weighting_teacher
        self.fwhm_student_teacher = fwhm_student_teacher
        self.fwhm_teacher_protos = fwhm_teacher_protos

    def forward(self,
                embeddings_students: List[torch.Tensor],
                embeddings_teacher: torch.Tensor,
                frames: torch.Tensor,
                coord_grids_students: List[torch.Tensor],
                coord_grids_teacher: torch.Tensor,
                temp_proto_teacher: float = 0.033,  # atm hardcoded
                temp_proto_student: float = 0.066,  # atm hardcoded
                dropout_rate: float = 0.2):
        """

        :param embeddings_students: [B, C, H, W, D]
        :param embeddings_teacher: [B, C, H, W, D]
        :param frames:
        :param coord_grids_students:
        :param coord_grids_teacher:
        :param dropout_rate:
        :return:
        """
        losses, plots = dict(), dict()
        device_ = embeddings_students[0].device
        n_batch, n_channels = embeddings_students[0].shape[0], embeddings_students[0].shape[1]
        n_students = len(embeddings_students)

        # Sample seeds for prototype clustering (atm on a grid)
        with torch.no_grad():
            theta = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]], device=device_).unsqueeze(0)
            reduced_size = [max(int(s_ // self.reduction_factor_protos), 1) for s_ in embeddings_teacher.shape[2:]]  # [3]
            affine_grids = F.affine_grid(theta=theta, size=[1, 1, *reduced_size], align_corners=False).expand(n_batch, -1, -1, -1, -1)  # [B, H', W', D', 3]
            embeddings_teacher_sampled = F.grid_sample(embeddings_teacher, grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)  # [B, C, H', W', D']
            coord_grids_teacher_sampled = F.grid_sample(coord_grids_teacher, grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)  # [B, 3, H', W', D']
            n_patch_sampled = math.prod(embeddings_teacher_sampled.shape[2:])

        # (Down-)sample student and teacher embeddings
        embeddings_students_reduced, coord_grids_students_reduced = [None for _ in range(len(embeddings_students))], [None for _ in range(len(embeddings_students))]
        embeddings_teacher_reduced, coord_grids_teacher_reduced = None, None
        for idx_emb, tuple_emb_grid_ in enumerate(zip(embeddings_students + [embeddings_teacher], coord_grids_students + [coord_grids_teacher])):
            with torch.no_grad():
                theta = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]], device=device_).unsqueeze(0)
                reduced_size = [max(int(s_ // self.reduction_factor), 1) for s_ in tuple_emb_grid_[0].shape[2:]]  # [3]
                affine_grids = F.affine_grid(theta=theta, size=[1, 1, *reduced_size], align_corners=False).expand(n_batch, -1, -1, -1, -1)  # [B, H', W', D', 3]
                spatial_jitter = torch.randint(low=0, high=int(math.ceil(self.reduction_factor)), size=(4,))
            if idx_emb < n_students:
                embeddings_ = tuple_emb_grid_[0][:, :, spatial_jitter[0]: tuple_emb_grid_[0].shape[2] - spatial_jitter[1], spatial_jitter[2]: tuple_emb_grid_[0].shape[3] - spatial_jitter[3], :]
                coord_grids_ = tuple_emb_grid_[1][:, :, spatial_jitter[0]: tuple_emb_grid_[1].shape[2] - spatial_jitter[1], spatial_jitter[2]: tuple_emb_grid_[1].shape[3] - spatial_jitter[3], :]
            else:
                embeddings_ = tuple_emb_grid_[0]
                coord_grids_ = tuple_emb_grid_[1]
            embeddings_ = F.grid_sample(embeddings_, grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)  # [B, C, H', W', D']
            coord_grids_ = F.grid_sample(coord_grids_, grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)
            if idx_emb < n_students:
                embeddings_students_reduced[idx_emb] = embeddings_
                coord_grids_students_reduced[idx_emb] = coord_grids_
            else:
                embeddings_teacher_reduced = embeddings_
                coord_grids_teacher_reduced = coord_grids_

        # Contrastive losses
        loss_sim_clustered = [torch.zeros((0,), device=device_) for _ in range(n_students)]
        unique_frames = np.unique(frames)
        for idx_frame, frame_ in enumerate(unique_frames):  # Note: This differentiation is not needed anymore, but may be helpful in the future for optional losses.
            valid_entries = np.array(frames) == frame_
            n_valid = np.count_nonzero(valid_entries)
            # if n_valid > 1:

            # Generate masks
            with torch.no_grad():
                pos_weights_student_teacher, indices_closest, mask_max_sim_dist = generate_masks_student_teacher(
                    coord_grids_student=[x_[valid_entries, ...] for x_ in coord_grids_students_reduced],
                    coord_grids_teacher=coord_grids_teacher_reduced[valid_entries, ...],
                    embedding_size=[x_.shape[2:] for x_ in embeddings_students_reduced],
                    embedding_size_teacher=embeddings_teacher_reduced.shape[2:],
                    fwhm=self.fwhm_student_teacher,
                )
                 # plots['pos_masks_student_proto'] = pos_weights_teacher_protos.reshape(n_valid, *embeddings_teacher.shape[2:], *embeddings_teacher_sampled.shape[2:])

            # Std contrastive learning to prototype targets
            # Retrieve proxy samples (that serve as proxy targets)
            # Note: we use a soft-assignment of prototypes chosen on a grid (as seeds / surrogates)
            with torch.no_grad():
                embeddings_teacher_valid = einops.rearrange(embeddings_teacher_reduced[valid_entries, ...], 'v c h w d -> v (h w d) c')  # [B, N, C]
                embeddings_teacher_valid_normed = F.normalize(embeddings_teacher_valid, p=2, dim=-1)
                embeddings_teacher_sampled_valid = einops.rearrange(embeddings_teacher_sampled[valid_entries, ...], 'v c h w d -> v (h w d) c')  # [V, P, C]
                embeddings_teacher_sampled_valid_normed = F.normalize(embeddings_teacher_sampled_valid, p=2, dim=-1)

                # Calc protos by soft k-means
                embeddings_protos_valid_normed = embeddings_teacher_sampled_valid_normed
                coords_protos = einops.rearrange(coord_grids_teacher_sampled[valid_entries, ...], 'v c h w d -> v (h w d) c')  # [V, P, 3]
                for idx_itr in range(self.k_means_iterations):
                    # Calc alignment
                    sim_emb_emb_teacher_protos = torch.einsum('v n c, v p c -> v n p', embeddings_teacher_valid_normed, embeddings_protos_valid_normed)  # [V, N, P]. Similarities between all teacher elements and sampled ones
                    sim_emb_emb_teacher_protos_soft = torch.softmax(sim_emb_emb_teacher_protos / temp_proto_teacher, dim=-1)  # Cluster alignment

                    # Calc position weights
                    pos_weights_teacher_protos = generate_masks_teacher_protos(
                        coord_grids_teacher=coord_grids_teacher_reduced[valid_entries, ...],
                        coord_grids_protos=coords_protos,
                        embedding_size_teacher=embeddings_teacher_reduced.shape[2:],
                        fwhm=self.fwhm_teacher_protos,
                    )
                    sim_emb_emb_teacher_teacher_sampled_soft_weighted = sim_emb_emb_teacher_protos_soft * pos_weights_teacher_protos if self.use_weighting_protos else sim_emb_emb_teacher_protos_soft

                    # Aggregate new protos and coords
                    embeddings_protos_valid = torch.einsum('v n p, v n c -> v p c', sim_emb_emb_teacher_teacher_sampled_soft_weighted, embeddings_teacher_valid)\
                                              / torch.sum(sim_emb_emb_teacher_teacher_sampled_soft_weighted, dim=1).unsqueeze(-1)  # [V, P, C] / [V, P, 1]. Denominator is not rly needed (if it is renormalized directly afterwards)
                    embeddings_protos_valid_normed = F.normalize(embeddings_protos_valid, p=2, dim=-1)  # [V, P, C]
                    coords_protos = torch.einsum('v n p, v n c -> v p c', sim_emb_emb_teacher_teacher_sampled_soft_weighted, einops.rearrange(coord_grids_teacher_reduced[valid_entries, ...], 'v c h w d -> v (h w d) c'))\
                                    / torch.sum(sim_emb_emb_teacher_teacher_sampled_soft_weighted, dim=1).unsqueeze(-1)  # [V, P, C] / [V, P, 1]

                # Recalc teacher proxy alignment
                sim_emb_emb_teacher_proxy = torch.einsum('v n c, v p c -> v n p', embeddings_teacher_valid_normed, embeddings_protos_valid_normed)
                sim_emb_emb_teacher_proxy_soft = torch.softmax(sim_emb_emb_teacher_proxy / temp_proto_teacher, dim=-1)  # [V, N, P]
                pos_weights_teacher_protos = generate_masks_teacher_protos(
                    coord_grids_teacher=coord_grids_teacher_reduced[valid_entries, ...],
                    coord_grids_protos=coords_protos,
                    embedding_size_teacher=embeddings_teacher_reduced.shape[2:],
                    fwhm=self.fwhm_teacher_protos,
                )
                sim_emb_emb_teacher_proxy_soft_final = sim_emb_emb_teacher_proxy_soft * pos_weights_teacher_protos if self.use_weighting_teacher else sim_emb_emb_teacher_proxy_soft
                if idx_frame == 0:
                    assignments = einops.rearrange(sim_emb_emb_teacher_proxy_soft_final, 'v (h w d) p -> v p h w d', h=embeddings_teacher_reduced.shape[2], w=embeddings_teacher_reduced.shape[3], d=embeddings_teacher_reduced.shape[4])
                    assignments_nonweighted = einops.rearrange(sim_emb_emb_teacher_proxy_soft, 'v (h w d) p -> v p h w d', h=embeddings_teacher_reduced.shape[2], w=embeddings_teacher_reduced.shape[3], d=embeddings_teacher_reduced.shape[4])
                    plots['sim_teacher_proxy'] = F.interpolate(assignments[..., assignments.shape[-1] // 2: assignments.shape[-1] // 2 + 1], scale_factor=(self.reduction_factor, self.reduction_factor, 1))
                    plots['sim_teacher_proxy_nonweighted'] = F.interpolate(assignments_nonweighted[..., assignments_nonweighted.shape[-1] // 2: assignments_nonweighted.shape[-1] // 2 + 1], scale_factor=(self.reduction_factor, self.reduction_factor, 1))

            for idx_student in range(n_students):
                embeddings_student_valid_ = einops.rearrange(embeddings_students_reduced[idx_student][valid_entries, ...], 'v c h w d -> v (h w d) c')  # [B, N, C]
                embeddings_student_valid_normed_ = F.normalize(embeddings_student_valid_, p=2, dim=-1)  # [B, C, H, W, D]. Normalize (for cosine sim)
                sim_emb_emb_student_proxy_ = torch.einsum('v n c, v p c -> v n p', embeddings_student_valid_normed_, embeddings_protos_valid_normed)  # [V, N, P]
                sim_emb_emb_student_proxy_soft_ = torch.softmax(sim_emb_emb_student_proxy_ / temp_proto_student, dim=-1)
                # sim_emb_emb_student_proxy_soft_weighted = sim_emb_emb_student_proxy_soft_ * pos_weights_student_protos if self.use_weighting else sim_emb_emb_student_proxy_soft_

                # Loss calc
                for idx_valid in range(n_valid):
                    if any(mask_max_sim_dist[idx_student][idx_valid, ...]):
                        sim_emb_emb_soft_selected_ = sim_emb_emb_student_proxy_soft_[idx_valid, ...][mask_max_sim_dist[idx_student][idx_valid, :], :]  # [N_sel, P]
                        cluster_assignments_selected_ = sim_emb_emb_teacher_proxy_soft_final[idx_valid, ...][indices_closest[idx_student][idx_valid, :], :][mask_max_sim_dist[idx_student][idx_valid, :], :]  # [N_sel, P]. Take the closest teacher->proxy assignment for each student position.
                        ce_clustered = - (cluster_assignments_selected_ * torch.clamp(torch.log(sim_emb_emb_soft_selected_ + 1e-16), min=-1e3, max=-0.)).sum(dim=1).mean(dim=0)
                        entropy_all = ce_clustered  # Currently, mean entropy isn't included.
                        loss_sim_clustered[idx_student] = torch.concat([loss_sim_clustered[idx_student], entropy_all.reshape(-1)])

        for idx_student in range(n_students):
            losses[f'contrastive_proxy_sim_clustered_s{idx_student}'] = self.loss_weight * loss_sim_clustered[idx_student].mean() if loss_sim_clustered[idx_student].shape[0] > 0 else torch.tensor(0., device=device_)

            with torch.no_grad():
                # Calculate plotted similarities - across non-pairs (for illustration)
                # Position is atm hardcoded
                embeddings_plot_candidates = embeddings_students[idx_student][...,
                                             embeddings_students[idx_student].shape[2] // 2 - 5: embeddings_students[idx_student].shape[2] // 2 + 5,
                                             embeddings_students[idx_student].shape[3] // 2 - 5: embeddings_students[idx_student].shape[3] // 2 + 5,
                                             # embeddings_students[idx_student].shape[2] // 4 - 5: embeddings_students[idx_student].shape[2] // 4 + 5,
                                             # int(embeddings_students[idx_student].shape[3] / 1.5 - 5): int(embeddings_students[idx_student].shape[3] / 1.5 + 5),
                                             :]
                embeddings_plot_candidates_shape = embeddings_plot_candidates.shape[2:]
                embeddings_plot_candidates = einops.rearrange(embeddings_plot_candidates, 'b c h w d -> b (h w d) c')
                embeddings_teacher = einops.rearrange(embeddings_teacher_reduced, 'b c h w d -> b (h w d) c')  # [B, N, C]
                embeddings_teacher_normed = F.normalize(embeddings_teacher, p=2, dim=-1)

                # Save exemplary similarities of student to teachers
                sim_emb_emb_student_proxy_plot = torch.einsum('b n c, b m c -> b n m', embeddings_plot_candidates, embeddings_teacher_normed)
                plots[f'sim_student_teacher_s{idx_student}'] = sim_emb_emb_student_proxy_plot.reshape(n_batch, *embeddings_plot_candidates_shape, *embeddings_teacher_reduced.shape[2:])  # Atm passes non-softmaxed similarities
        return losses, plots


def generate_masks_student_teacher(coord_grids_student: List[torch.Tensor],
                                   coord_grids_teacher: torch.Tensor,
                                   embedding_size: List[Tuple[int, int, int]],
                                   embedding_size_teacher: Tuple[int, int, int],
                                   fwhm: float = 256.,
                                   max_sim_dist: Tuple[float, ...] = (4., 2.),
                                   scale_z: float = 2.0,
                                   thresh: float = 0.5):

    # x,y,z position differences - student to teacher
    pos_masks_student_teacher = list()
    indices_closest = list()
    mask_max_sim_dist = list()
    coord_grids_teacher_zoomed = F.interpolate(coord_grids_teacher, size=embedding_size_teacher, mode='trilinear')  # resize to feature map size
    for idx_student in range(len(coord_grids_student)):
        coord_grids_student_zoomed = F.interpolate(coord_grids_student[idx_student], size=embedding_size[idx_student], mode='trilinear')  # resize to feature map size
        diff_xyz = (einops.rearrange(coord_grids_student_zoomed, 'b c h w d -> c b (h w d) ()') - einops.rearrange(coord_grids_teacher_zoomed, 'b c h w d -> c b () (h w d)'))  # [3, B, N1, N2].
        diff_xyz[2, ...] *= scale_z  # penalize coord diff in z-direction more strongly.
        diff_all = torch.linalg.norm(diff_xyz[:3, ...], ord=2, dim=0)  # [B, N1, N2]

        sigma_squared = (fwhm / 2.355)**2  # FWHM ~= 2.355*sigma
        pos_masks_student_teacher.append(torch.exp(- diff_all ** 2 / (2 * sigma_squared)) >= thresh)  # [B, N1, N2]. Weights are compared to threshold to produce binary mask.
        pos_minimum, indices_closest_ = torch.min(diff_all, dim=-1)  # [B, N1], [B, N1].
        indices_closest.append(indices_closest_)
        mask_max_sim_dist.append(pos_minimum <= max_sim_dist[0])  # [B, N1].

    return pos_masks_student_teacher, indices_closest, mask_max_sim_dist


def generate_masks_teacher_protos(coord_grids_teacher: torch.Tensor,
                                  coord_grids_protos: torch.Tensor,
                                  embedding_size_teacher: Tuple[int, int, int],
                                  fwhm: float = 256.,
                                  scale_z: float = 2.0):
    coord_grids_teacher_zoomed = F.interpolate(coord_grids_teacher, size=embedding_size_teacher, mode='trilinear')  # resize to feature map size

    # x,y,z position differences - teacher to prototype surrogates
    diff_xyz = (einops.rearrange(coord_grids_teacher_zoomed, 'b c h w d -> c b (h w d) ()') - einops.rearrange(coord_grids_protos, 'b n c -> c b () n'))  # [3, B, N1, N2]. Protos are already in node shape.
    diff_xyz[2, ...] *= scale_z  # penalize coord diff in z-direction more strongly.
    diff_all = torch.linalg.norm(diff_xyz[:3, ...], ord=2, dim=0)  # [B, N1, N2]

    sigma_squared = (fwhm / 2.355)**2  # FWHM ~= 2.355*sigma
    pos_weights_teacher_protos = torch.exp(- diff_all ** 2 / (2 * sigma_squared))  # [B, N1, N2]. True weights (not masks).

    return pos_weights_teacher_protos


