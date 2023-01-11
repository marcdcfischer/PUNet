from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Any, Dict, Union, List, Tuple
import pathlib as plb
from src.modules.architectures.momentum_model import MomentumModel
from src.modules.losses.contrastive_protos_teacher import ContrastiveProtosTeacherLoss
from src.utils.plotting import image_grid, similarities_student_teacher
from src.modules.losses import focal
import wandb
import monai
import monai.inferers as minferers
from monai.data import decollate_batch
import einops
import torch.nn.functional as F
import itertools
from functools import partial
import numpy as np
from src.data.transforms_monai import generate_test_post_transforms
import nibabel as nib
import warnings


class InstructionModel(pl.LightningModule):
    def __init__(self, conf: Union[Dict, Namespace], **kwargs):
        super().__init__()
        self.save_hyperparameters(conf)

        self.pseudo_indices_subject_idx = None
        self.pseudo_indices_label_idx = None
        self.label_indices_active = None
        self.label_indices_frozen = None
        self.mode_loss = None

        print(f'Establishing architecture with parameters: \n {self.hparams}')

        # Architecture
        self.architecture = MomentumModel(conf=self.hparams)

        # Losses
        # Segmentation losses
        self.loss_seg = focal.FocalLoss(self.hparams.out_channels,
                                        loss_weight=self.hparams.loss_weight_segmentation,
                                        gamma=self.hparams.loss_weight_segmentation_gamma,
                                        alpha_background=self.hparams.loss_weight_segmentation_alpha_background if not self.hparams.downstream else self.hparams.loss_weight_segmentation_alpha_background_downstream,
                                        alpha_foreground=self.hparams.loss_weight_segmentation_alpha_foreground,
                                        additive_alpha=self.hparams.additive_alpha)

        # Contrastive losses
        self.loss_cluster_pairs = ContrastiveProtosTeacherLoss(
            reduction_factor=self.hparams.reduction_factor,
            reduction_factor_protos=self.hparams.reduction_factor_protos,
            loss_weight=self.hparams.loss_weight_sim_protos if not self.hparams.downstream else self.hparams.loss_weight_sim_protos_downstream,
            k_means_iterations=self.hparams.k_means_iterations,
            use_weighting_protos=self.hparams.use_weighting_protos,
            use_weighting_teacher=self.hparams.use_weighting_teacher,
            fwhm_student_teacher=self.hparams.fwhm_student_teacher,
            fwhm_teacher_protos=self.hparams.fwhm_teacher_protos,
        )

        # Metrics
        self.score_seg_train = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_train_annotated = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_train_non_annotated = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_val = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_test = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')

    def forward(self,
                x: List[torch.Tensor],
                x_teacher: Optional[torch.Tensor] = None,
                label_indices: Optional[torch.Tensor] = None,
                pseudo_indices_subject: Optional[torch.Tensor] = None,
                pseudo_indices_label: Optional[torch.Tensor] = None,
                mode_label: str = 'label',
                mode_loss: str = 'both',
                teacher_prediction: bool = True,
                second_student_prediction: bool = True):

        x_teacher = x_teacher if teacher_prediction else None
        x = x if second_student_prediction else x[:1]
        dict_out_students, dict_out_teacher = self.architecture(x, x_teacher, label_indices, pseudo_indices_subject, pseudo_indices_label,
                                                                mode_label=mode_label, mode_loss=mode_loss)

        return dict_out_students, dict_out_teacher

    def forward_prediction(self, x: torch.Tensor, label_indices: torch.Tensor):  # takes an image and predicts one (as required for monai sliding window inference)

        label_indices = label_indices.expand(x.shape[0], -1)  # Expand instructions to all patches (from the sliding window)
        dict_out_students, _ = self.architecture([x], None, label_indices)

        return dict_out_students[0]['dense']['embedded_latents']

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if batch_idx == 0:
            torch.cuda.empty_cache()  # helps with fragmentation
            if self.current_epoch == 0:
                print(f'Training with optimizer(s) {self.optimizers()}')
            if self.hparams.downstream and self.lr_schedulers() is not None:
                print(f'Training epoch {self.current_epoch} with step size(s) {self.lr_schedulers().get_last_lr()}')
                for idx_step_size, step_size_ in enumerate(self.lr_schedulers().get_last_lr()):
                    self.log(f'train_step_size_{idx_step_size}', step_size_, sync_dist=True)

        # batch selection
        if self.hparams.orientation_swap:
            batch = batch[batch_idx % len(batch)]  # take only one of the loaded batches per batch_idx
        else:
            batch = batch[0]

        self.mode_loss = 'both'

        # Batch preparations
        # Aux content for one of the categories (student large) only
        aux_names = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'name' in str(x_)]))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        aux_frames = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'frame' in str(x_)]))))
        aux_domains = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'domain' in str(x_)]))))
        aux_annotated = torch.stack(list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'annotated' in str(x_)])))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.

        # Image content
        x_students = [einops.rearrange(batch['image'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms)]  # Concatenate differently transformed elements in batch dim
        y_students = [einops.rearrange(batch['label'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms).float().round().long()[:, 0, ...]]
        for idx_student in range(self.hparams.n_students - 1):
            x_students.append(einops.rearrange(batch[f'image_var{idx_student}'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms))
            y_students.append(einops.rearrange(batch[f'label_var{idx_student}'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms).float().round().long()[:, 0, ...])
        x_teacher = einops.rearrange(batch['image_teacher'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms)
        y_teacher = einops.rearrange(batch['label_teacher'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms).float().round().long()[:, 0, ...]

        y_students_one_hot = [einops.rearrange(F.one_hot(x_, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d') for x_ in y_students]
        y_teacher_one_hot = einops.rearrange(F.one_hot(y_teacher, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d')

        coord_grids_students = None
        coord_grids_teacher = None
        if 'coord_grid' in batch.keys():
            coord_grids_students = [einops.rearrange(batch['coord_grid'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms)]
            for idx_student in range(self.hparams.n_students - 1):
                coord_grids_students.append(einops.rearrange(batch[f'coord_grid_var{idx_student}'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms))
            coord_grids_teacher = einops.rearrange(batch['coord_grid_teacher'], 'b (t c) d h w -> (b t) c d h w', t=self.hparams.num_transforms)

        # Switch label indices (and thereby instructions)
        self.set_active_indices_training(batch_size=x_students[0].shape[0], device=y_students[0].device)
        self.set_frozen_indices(device=y_students[0].device)
        y_one_hot_gathered, y_gathered, y_one_hot_active = list(zip(*[self.gather_labels(targets_one_hot=x_) for x_ in y_students_one_hot]))
        y_teacher_one_hot_gathered, y_teacher_gathered, y_teacher_one_hot_active = self.gather_labels(targets_one_hot=y_teacher_one_hot)

        # Enable / disable learning for bulk of interpreter
        if self.hparams.selective_freezing:
            self.selective_freezing(batch_idx=batch_idx)

        # Momentum update
        self.architecture.update_teacher()

        # prediction
        dict_out_students, dict_out_teacher = self(x_students,
                                                   x_teacher,
                                                   label_indices=self.label_indices_active,
                                                   pseudo_indices_subject=self.pseudo_indices_subject_idx,
                                                   pseudo_indices_label=self.pseudo_indices_label_idx,
                                                   mode_label='label',
                                                   mode_loss=self.mode_loss,
                                                   teacher_prediction=self.loss_cluster_pairs.loss_weight > 0.,
                                                   second_student_prediction=self.loss_cluster_pairs.loss_weight > 0.)

        # Loss calculation
        loss_dict, plots_dict = dict(), dict()
        # Segmentation
        for idx_student in range(len(dict_out_students)):
            if (self.mode_loss == 'seg' or self.mode_loss == 'both') and self.hparams.loss_weight_segmentation > 0.:  # any(aux_annotated):
                if not self.hparams.downstream:
                    loss_dict.update(self.loss_seg(dict_out_students[idx_student]['dense']['embedded_latents'][aux_annotated, ...],
                                                   y_gathered[idx_student][aux_annotated, ...],
                                                   label_indices_active=self.label_indices_active,
                                                   tag=f'seg_s{idx_student}'))
                else:
                    dense_embedded_latents_scattered = self.scatter_predictions(dict_out_students[idx_student]['dense']['embedded_latents'])
                    dense_embedded_latents_nonfrozen = self.gather_non_frozen_predictions(dense_embedded_latents_scattered)
                    y_one_hot_non_frozen_, y_non_frozen_ = self.gather_non_frozen_labels(y_students_one_hot[idx_student])
                    loss_dict.update(self.loss_seg(dense_embedded_latents_nonfrozen[aux_annotated, ...],
                                                   y_non_frozen_[aux_annotated, ...],
                                                   label_indices_active=self.label_indices_active,
                                                   tag=f'seg_s{idx_student}'))
            else:
                loss_dict[f'seg_s{idx_student}'] = torch.tensor(0., device=self.device, requires_grad=False)
            loss_dict[f'seg_pseudo_s{idx_student}'] = torch.tensor(0., device=self.device, requires_grad=False)

        # Self-supervised contrastive
        if self.hparams.contrastive_pairs:
            if (self.mode_loss == 'self' or self.mode_loss == 'both') and self.loss_cluster_pairs.loss_weight > 0.:
                loss_tmp_, plots_tmp_ = self.loss_cluster_pairs(embeddings_students=[x_['patched']['embedded_latents'] for x_ in dict_out_students],
                                                                embeddings_teacher=dict_out_teacher['patched']['embedded_latents'],
                                                                frames=aux_frames,
                                                                coord_grids_students=coord_grids_students,
                                                                coord_grids_teacher=coord_grids_teacher)
                loss_dict.update(loss_tmp_), plots_dict.update(plots_tmp_)
            else:
                for idx_student in range(len(dict_out_students)):
                    loss_dict[f'contrastive_proxy_sim_clustered_s{idx_student}'] = torch.tensor(0., device=self.device, requires_grad=False)
        loss_dict['all'] = sum(loss_dict.values())

        with torch.no_grad():
            # metrics
            for idx_student in range(len(dict_out_students)):
                dense_embedded_latents_scattered_ = self.scatter_predictions(dict_out_students[idx_student]['dense']['embedded_latents'])
                predictions_one_hot_ = F.one_hot(dense_embedded_latents_scattered_.argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
                self.score_seg_train(y_pred=predictions_one_hot_,
                                     y=y_one_hot_active[idx_student])
                if any(aux_annotated):
                    self.score_seg_train_annotated(y_pred=predictions_one_hot_[aux_annotated, ...],
                                                   y=y_one_hot_active[idx_student][aux_annotated, ...])
                if any(~aux_annotated):
                    self.score_seg_train_non_annotated(y_pred=predictions_one_hot_[~aux_annotated, ...],
                                                       y=y_one_hot_active[idx_student][~aux_annotated, ...])

            # logging
            for key_, value_ in loss_dict.items():
                self.log(f'train_loss_{key_}', value_.detach().cpu())

            # plotting
            if self.hparams.plot and (self.current_epoch + 1) % self.hparams.plot_interval_train == 0 and batch_idx % 10 == 0 and batch_idx <= 20:
                if not dist.is_initialized() or dist.get_rank() < 1:
                    # Fetch valid elements (if sim of plots has only been calculated for valid elements)
                    valid_entries = np.array(aux_frames) == np.unique(aux_frames)[0]
                    png_paths = list()
                    for idx_student in range(len(dict_out_students)):
                        png_paths.extend(similarities_student_teacher.visualize_similarities_student_teacher({k_: v_ for k_, v_ in plots_dict.items() if f'{idx_student}' in k_}, x_students[idx_student], x_teacher, y_teacher,
                                                                                                             prefix=f'{self.hparams.run_name}_train_b{batch_idx}_s{idx_student}_ep{str(self.current_epoch).zfill(3)}', path_plots=self.hparams.default_root_dir))
                        y_one_hot_list_ = [y_one_hot_gathered[idx_student], y_teacher_one_hot_gathered]
                        for identifier_, dict_, x_, y_one_hot_, in zip(['student', 'teacher'], [dict_out_students[idx_student], dict_out_teacher], [x_students[idx_student], x_teacher], y_one_hot_list_):
                            if dict_ is not None:
                                png_paths.extend(image_grid.plot_grid_middle(x_.detach().cpu(),
                                                                             y_one_hot_.detach().cpu(),
                                                                             torch.softmax(dict_['dense']['embedded_latents'].detach().cpu().float(), dim=1),
                                                                             None,
                                                                             indices_elements=[idx_ * 2 for idx_ in range(max(min(x_students[0].shape[0] // 2, 2), 1))],
                                                                             prefix=f'{self.hparams.run_name}_train_b{batch_idx}_s{idx_student}_ep{str(self.current_epoch).zfill(3)}_{identifier_}', path_plots=self.hparams.default_root_dir))
                    if self.hparams.online_on:
                        [self.logger[1].experiment.log({str(plb.Path(png_).stem[:-6]): wandb.Image(png_)}) for png_ in png_paths]

        return {'loss': loss_dict['all']}

    def training_epoch_end(self, outputs) -> None:
        self.log(f'train_loss_mean', torch.mean(torch.tensor([x_['loss'] for x_ in outputs], dtype=torch.float)), sync_dist=True)
        for scorer_, tag_ in zip([self.score_seg_train], ['all']):  # self.score_seg_train_annotated, self.score_seg_train_non_annotated], ['all', 'annotated', 'non_annotated']):
            try:
                score_seg = scorer_.aggregate()
                scorer_.reset()
                self.log_dict({f'train_dice_{tag_}_c{str(idx_c).zfill(2)}': score_seg[idx_c] for idx_c in range(len(score_seg))}, sync_dist=True)
                self.log(f'train_dice_{tag_}_mean', torch.mean(score_seg[1:]), sync_dist=True)
            except:
                pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if batch_idx == 0:
            torch.cuda.empty_cache()  # helps with fragmentation

        # batch preparations
        aux_names = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'name' in x_]))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        aux_frames = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'frame' in x_]))))
        aux_domains = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'domain' in x_]))))
        aux_annotated = torch.stack(list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'annotated' in x_])))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        x = batch['image']
        y = batch['label'].float().round().long()[:, 0, ...]
        y_one_hot = einops.rearrange(F.one_hot(y, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d')
        scribbles, scribbles_masked = None, None
        if 'scribbles' in batch.keys():
            scribbles = batch['scribbles'][:, 0, ...]
            scribbles_masked = scribbles.clone()
            scribbles_masked[~aux_annotated, ...] = self.hparams.out_channels  # Mask non-annotated
        coord_grids = None
        if 'coord_grid' in batch.keys():
            coord_grids = batch['coord_grid']

        # Activate respective instructions
        loss_dict_all = list()
        foreground_labels = self.hparams.label_indices_downstream_active if self.hparams.downstream else self.hparams.label_indices_base
        foreground_labels = foreground_labels if self.hparams.downstream else []
        for idx_category in range(len(foreground_labels)):
            self.set_active_indices_validation(batch_size=x.shape[0], foreground_category=foreground_labels[idx_category], device=y.device)
            self.set_frozen_indices(device=y.device)
            y_one_hot_gathered, y_gathered, y_one_hot_active = self.gather_labels(targets_one_hot=y_one_hot)

            # Sliding Window Inference - by teacher
            print(f'Evaluating batch {batch_idx} for category {foreground_labels[idx_category]} ({idx_category + 1}/{len(foreground_labels)}) with active indices {self.label_indices_active}.')
            volume_prediction = minferers.sliding_window_inference(inputs=x,
                                                                   roi_size=self.hparams.patch_size_students[0],
                                                                   sw_batch_size=self.hparams.batch_size,
                                                                   predictor=partial(self.forward_prediction, label_indices=self.label_indices_active),
                                                                   overlap=self.hparams.sliding_window_overlap)

            # Loss calculation
            loss_dict, plots_dict = dict(), dict()
            # Segmentation
            if any(aux_annotated):
                if not self.hparams.downstream and self.hparams.loss_weight_segmentation > 0.:
                    loss_dict.update(self.loss_seg(volume_prediction[aux_annotated, ...],
                                                   y_gathered[aux_annotated, ...],
                                                   label_indices_active=self.label_indices_active))
                else:
                    dense_embedded_latents_scattered = self.scatter_predictions(volume_prediction)
                    dense_embedded_latents_nonfrozen = self.gather_non_frozen_predictions(dense_embedded_latents_scattered)
                    y_one_hot_non_frozen_, y_non_frozen_ = self.gather_non_frozen_labels(y_one_hot)
                    loss_dict.update(self.loss_seg(dense_embedded_latents_nonfrozen[aux_annotated, ...],
                                                   y_non_frozen_[aux_annotated, ...],
                                                   label_indices_active=self.label_indices_active))
            else:
                loss_dict['seg'] = torch.tensor(0., device=self.device, requires_grad=False)
            # Self-supervised contrastive not available in the current setup, since patched embeddings would be required
            loss_dict['all'] = sum(loss_dict.values())
            loss_dict_all.append(loss_dict)

            with torch.no_grad():
                # metrics
                dense_embedded_latents_scattered = self.scatter_predictions(volume_prediction)
                predictions_one_hot = F.one_hot(dense_embedded_latents_scattered.argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
                self.score_seg_val(y_pred=predictions_one_hot,
                                   y=y_one_hot_active)

                # plotting
                if self.hparams.plot and (self.current_epoch + 1) % self.hparams.plot_interval_val == 0 and batch_idx % 10 == 0 and batch_idx <= 20:
                    if not dist.is_initialized() or dist.get_rank() < 1:
                        png_paths = list()
                        png_paths.extend(image_grid.plot_grid_middle(x.detach().cpu(),
                                                                     y_one_hot_gathered.detach().cpu(),
                                                                     torch.softmax(volume_prediction.detach().cpu(), dim=1),
                                                                     scribbles.detach().cpu() if scribbles is not None else None,
                                                                     indices_elements=[idx_ * 2 for idx_ in range(max(min(x.shape[0] // 2, 2), 1))],
                                                                     prefix=f'{self.hparams.run_name}_val_b{batch_idx}_s0_c{idx_category}_ep{str(self.current_epoch).zfill(3)}', path_plots=self.hparams.default_root_dir))
                        if self.hparams.online_on:
                            [self.logger[1].experiment.log({str(plb.Path(png_).stem[:-6]): wandb.Image(png_)}) for png_ in png_paths]

        if len(loss_dict_all) > 0:
            loss_dict_all = {k_: sum(dict_[k_] for dict_ in loss_dict_all) / len(loss_dict_all) for k_ in loss_dict_all[0].keys()}

            with torch.no_grad():
                # logging
                self.log(f'val_epoch', self.current_epoch, sync_dist=True)  # log for ckpt selection
                for key_, value_ in loss_dict_all.items():
                    self.log(f'val_loss_{key_}', value_, sync_dist=True)  # logs per epoch
        else:
            loss_dict_all = {'all': torch.tensor(0., device=self.device, requires_grad=False)}

        return {'loss': loss_dict_all['all']}

    def validation_epoch_end(self, outputs) -> None:
        self.log(f'val_loss_mean', torch.mean(torch.tensor([x_['loss'] for x_ in outputs], dtype=torch.float)), sync_dist=True)
        if self.hparams.downstream:
            score_seg = self.score_seg_val.aggregate()
            self.score_seg_val.reset()
            self.log_dict({f'val_dice_c{str(idx_c).zfill(2)}': score_seg[idx_c] for idx_c in range(len(score_seg))}, sync_dist=True)
            self.log(f'val_dice_mean', torch.mean(score_seg[1:]), sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if batch_idx == 0:
            torch.cuda.empty_cache()  # helps with fragmentation

        # batch preparations
        aux_names = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'name' in x_]))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        aux_frames = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'frame' in x_]))))
        aux_domains = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'domain' in x_]))))
        aux_annotated = torch.stack(list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'annotated' in x_])))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        x = batch['image']
        y = batch['label'].float().round().long()[:, 0, ...]
        y_one_hot = einops.rearrange(F.one_hot(y, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d')
        scribbles, scribbles_masked = None, None
        if 'scribbles' in batch.keys():
            scribbles = batch['scribbles'][:, 0, ...]
            scribbles_masked = scribbles.clone()
            scribbles_masked[~aux_annotated, ...] = self.hparams.out_channels  # Mask non-annotated
        coord_grids = None
        if 'coord_grid' in batch.keys():
            coord_grids = batch['coord_grid']

        # Activate respective instructions
        loss_dict_all = list()
        foreground_labels = self.hparams.label_indices_downstream_active if self.hparams.downstream else self.hparams.label_indices_base
        for idx_category in range(len(foreground_labels)):
            torch.cuda.empty_cache()  # helps with fragmentation
            self.set_active_indices_test(batch_size=x.shape[0], foreground_category=foreground_labels[idx_category], device=y.device)
            self.set_frozen_indices(device=y.device)
            y_one_hot_gathered, y_gathered, y_one_hot_active = self.gather_labels(targets_one_hot=y_one_hot)

            # Sliding Window Inference - by teacher
            print(f'Evaluating batch {batch_idx} for category {foreground_labels[idx_category]} ({idx_category + 1}/{len(foreground_labels)}) with active indices {self.label_indices_active}.')
            volume_prediction = minferers.sliding_window_inference(inputs=x,
                                                                   roi_size=self.hparams.patch_size_students[0],
                                                                   sw_batch_size=self.hparams.batch_size,
                                                                   predictor=partial(self.forward_prediction, label_indices=self.label_indices_active),
                                                                   overlap=self.hparams.sliding_window_overlap)

            # Loss calculation
            loss_dict, plots_dict = dict(), dict()
            # Segmentation
            test_loss = False
            if test_loss and any(aux_annotated):
                if not self.hparams.downstream and self.hparams.loss_weight_segmentation > 0.:
                    loss_dict.update(self.loss_seg(volume_prediction[aux_annotated, ...],
                                                   y_gathered[aux_annotated, ...],
                                                   label_indices_active=self.label_indices_active))
                else:
                    dense_embedded_latents_scattered = self.scatter_predictions(volume_prediction)
                    dense_embedded_latents_nonfrozen = self.gather_non_frozen_predictions(dense_embedded_latents_scattered)
                    y_one_hot_non_frozen_, y_non_frozen_ = self.gather_non_frozen_labels(y_one_hot)
                    loss_dict.update(self.loss_seg(dense_embedded_latents_nonfrozen[aux_annotated, ...],
                                                   y_non_frozen_[aux_annotated, ...],
                                                   label_indices_active=self.label_indices_active))
            else:
                loss_dict['seg'] = torch.tensor(0., device=self.device, requires_grad=False)
            # Self-supervised contrastive not available in the current setup, since patched embeddings would be required
            loss_dict['all'] = sum(loss_dict.values())
            loss_dict_all.append(loss_dict)

            with torch.no_grad():
                # metrics
                dense_embedded_latents_scattered = self.scatter_predictions(volume_prediction)
                test_score = False
                if test_score:
                    predictions_one_hot = F.one_hot(dense_embedded_latents_scattered.argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
                    self.score_seg_test(y_pred=predictions_one_hot,
                                        y=y_one_hot_active)

                # save predictions
                test_viz = False
                if test_viz:
                    import matplotlib
                    matplotlib.use('tkagg')
                    argmaxed = torch.argmax(dense_embedded_latents_scattered[0, ...], dim=0)
                    viewer = nib.viewers.OrthoSlicer3D(np.array((argmaxed / argmaxed.max()).detach().cpu()))
                    viewer.clim = [0.0, 1.0]
                    viewer.show()
                post_transform = generate_test_post_transforms(output_dir=self.hparams.export_dir,
                                                               output_postfix=f'pred_cat{foreground_labels[idx_category]}',
                                                               transform_test=self.trainer.datamodule.transform_test,
                                                               n_classes=None)
                batch['pred'] = dense_embedded_latents_scattered
                [post_transform(x_) for x_ in decollate_batch(batch)]

                # plotting
                if self.hparams.plot and self.hparams.plot_interval_test > 0 and batch_idx % 10 == 0 and batch_idx <= 20:
                    if not dist.is_initialized() or dist.get_rank() < 1:
                        png_paths = list()
                        png_paths.extend(image_grid.plot_grid_middle(x.detach().cpu(),
                                                                     y_one_hot_gathered.detach().cpu(),
                                                                     torch.softmax(volume_prediction.detach().cpu(), dim=1),
                                                                     scribbles.detach().cpu() if scribbles is not None else None,
                                                                     indices_elements=[idx_ * 2 for idx_ in range(max(min(x.shape[0] // 2, 2), 1))],
                                                                     prefix=f'{self.hparams.run_name}_test_b{batch_idx}_s0_c{idx_category}_ep{str(self.current_epoch).zfill(3)}', path_plots=self.hparams.default_root_dir))
                        if self.hparams.online_on:
                            [self.logger[1].experiment.log({str(plb.Path(png_).stem[:-6]): wandb.Image(png_)}) for png_ in png_paths]

        if len(loss_dict_all) > 0:
            loss_dict_all = {k_: sum(dict_[k_] for dict_ in loss_dict_all) / len(loss_dict_all) for k_ in loss_dict_all[0].keys()}

            with torch.no_grad():
                # logging
                self.log(f'test_epoch', self.current_epoch, sync_dist=True)  # log for ckpt selection
                for key_, value_ in loss_dict_all.items():
                    self.log(f'test_loss_{key_}', value_, sync_dist=True)  # logs per epoch
        else:
            loss_dict_all = {'all': torch.tensor(0., device=self.device, requires_grad=False)}

        return {'loss': loss_dict_all['all']}

    def test_epoch_end(self, outputs) -> None:
        self.log(f'test_loss_mean', torch.mean(torch.tensor([x_['loss'] for x_ in outputs], dtype=torch.float)), sync_dist=True)
        test_score = False
        if test_score:
            score_seg = self.score_seg_test.aggregate()
            self.score_seg_test.reset()
            self.log_dict({f'test_dice_c{str(idx_c).zfill(2)}': score_seg[idx_c] for idx_c in range(len(score_seg))}, sync_dist=True)
            self.log(f'test_dice_mean', torch.mean(score_seg[1:]), sync_dist=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        pass

    def configure_optimizers(self):
        optimizer = AdamW([{'params': [param_ for name_, param_ in [*self.architecture.network_student.get_named_body_parameters(), *self.architecture.network_student.get_named_adapter_parameters()]]},
                           {'params': [param_ for name_, param_ in self.architecture.network_student.get_named_instruction_parameters()],
                            'lr': self.hparams.learning_rate_instructions if not self.hparams.downstream else self.hparams.learning_rate_instructions_downstream}],
                          lr=self.hparams.learning_rate if not self.hparams.downstream else self.hparams.learning_rate_downstream,
                          weight_decay=self.hparams.weight_decay if not self.hparams.downstream else self.hparams.weight_decay_downstream)

        if self.hparams.downstream and self.hparams.with_scheduler_downstream:
            assert self.hparams.max_epochs is not None
            print(f'Using one cycle lr scheduler for {self.hparams.max_epochs} epochs and {1} steps per epoch.')
            scheduler = lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=[self.hparams.learning_rate_downstream, self.hparams.learning_rate_instructions_downstream],
                                                total_steps=None,
                                                epochs=self.hparams.max_epochs,
                                                steps_per_epoch=1,  # The amount of scheduler.step() performed in an epoch. Probably defaults to 1 for lightning.
                                                pct_start=0.1,
                                                anneal_strategy='cos',
                                                cycle_momentum=True,
                                                base_momentum=0.85,
                                                max_momentum=0.95,
                                                div_factor=20,  # 1e-2 / 1e2 = 1e-4
                                                final_div_factor=1,  # 1e-4 / 1 = 1e-4
                                                three_phase=False,
                                                last_epoch=- 1,
                                                verbose=False)
            return [optimizer], [scheduler]
        return optimizer

    def selective_freezing(self, batch_idx):
        if batch_idx == 0:
            grad_instructions = self.label_indices_active[0].bool() if self.hparams.freeze_inactive else ~self.label_indices_frozen.bool()  # Expects all batch entries to have the same target during downstream
            if self.hparams.separate_background:
                grad_instructions[0] = False  # 0 isn't used in this case in the pools anyway.

            print(f'\nBatch idx: {batch_idx} - Active labels: {self.label_indices_active[0].bool()}.')
            print(f'\nBatch idx: {batch_idx} - Gradients for instructions: {grad_instructions},'
                  f' instructions norm: {not self.hparams.freeze_body},'
                  f' instruction bias scores: {not self.hparams.freeze_bias_scores},'
                  f' body: {not self.hparams.freeze_norm}.')
            self.architecture.network_student.set_requires_gradient(grad_instructions=grad_instructions,
                                                                    grad_instructions_norm=not self.hparams.freeze_norm,
                                                                    grad_instructions_scores=not self.hparams.freeze_bias_scores,
                                                                    grad_body=not self.hparams.freeze_body)

    def set_active_indices_training(self,
                                    batch_size: int,
                                    device: torch.device):
        if not self.hparams.downstream:
            # Active categories
            active_classes = torch.randint(low=self.hparams.label_indices_min_active, high=self.hparams.label_indices_max_active + 1, size=(1,), device=device)[0]  # Atm same active number of classes across a batch (otherwise some kind of masking would be needed in the att.)
            weights = torch.ones((len(self.hparams.label_indices_base),), dtype=torch.float, device=device).reshape(1, -1).expand(batch_size, -1)  # [B, L]
            rand_indices = torch.multinomial(weights, num_samples=active_classes, replacement=False)
            rand_foreground = torch.tensor(self.hparams.label_indices_base, dtype=torch.long, device=device)[rand_indices]
            active_indices = torch.zeros((batch_size, self.hparams.out_channels), dtype=torch.long, device=device)
            active_indices[:, 0] = 1  # Set background
            for idx_active in range(active_classes):  # Set foreground
                active_indices[torch.arange(batch_size), rand_foreground[:, idx_active]] = 1
            self.label_indices_active = active_indices
        else:
            # Active categories
            active_indices = torch.zeros((batch_size, self.hparams.out_channels), dtype=torch.long, device=device)
            active_indices[:, 0] = 1
            for idx_active in range(len(self.hparams.label_indices_downstream_active)):
                active_indices[:, self.hparams.label_indices_downstream_active[idx_active]] = 1
            self.label_indices_active = active_indices

    def set_active_indices_validation(self,
                                      batch_size: int,
                                      foreground_category: int,  # categories can be drawn manually (alongside random ones)
                                      device: torch.device):
        if not self.hparams.downstream:
            active_classes = self.hparams.label_indices_max_active
            weights = torch.ones((self.hparams.out_channels - 1,), dtype=torch.float, device=device).reshape(1, -1).expand(batch_size, -1)
            weights[:, foreground_category - 1] = 0.
            rand_indices = torch.tensor([[foreground_category - 1]], dtype=torch.long, device=device)
            if active_classes > 1:
                rand_indices = torch.concat([rand_indices, torch.multinomial(weights, num_samples=active_classes - 1, replacement=False)], dim=1)
            rand_foreground = torch.arange(1, self.hparams.out_channels, dtype=torch.long, device=device)[rand_indices]
            active_indices = torch.zeros((batch_size, self.hparams.out_channels), dtype=torch.long, device=device)
            active_indices[:, 0] = 1  # Set background
            for idx_active in range(active_classes):  # Set foreground
                active_indices[torch.arange(batch_size), rand_foreground[:, idx_active]] = 1
            self.label_indices_active = active_indices
        else:
            # Active categories
            active_indices = torch.zeros((batch_size, self.hparams.out_channels), dtype=torch.long, device=device)
            active_indices[:, 0] = 1
            for idx_active in range(len(self.hparams.label_indices_downstream_active)):
                active_indices[:, self.hparams.label_indices_downstream_active[idx_active]] = 1
            self.label_indices_active = active_indices

    def set_active_indices_test(self,
                                batch_size: int,
                                foreground_category: int,  # categories can be drawn manually (alongside random ones)
                                device: torch.device):
        if not self.hparams.downstream:
            active_classes = self.hparams.label_indices_max_active
            weights = torch.ones((self.hparams.out_channels - 1,), dtype=torch.float, device=device).reshape(1, -1).expand(batch_size, -1)
            weights[:, foreground_category - 1] = 0.
            rand_indices = torch.tensor([[foreground_category - 1]], dtype=torch.long, device=device)
            if active_classes > 1:
                rand_indices = torch.concat([rand_indices, torch.multinomial(weights, num_samples=active_classes - 1, replacement=False)], dim=1)
            rand_foreground = torch.arange(1, self.hparams.out_channels, dtype=torch.long, device=device)[rand_indices]
            active_indices = torch.zeros((batch_size, self.hparams.out_channels), dtype=torch.long, device=device)
            active_indices[:, 0] = 1  # Set background
            for idx_active in range(active_classes):  # Set foreground
                active_indices[torch.arange(batch_size), rand_foreground[:, idx_active]] = 1
            self.label_indices_active = active_indices
        else:
            # Active categories
            active_indices = torch.zeros((batch_size, self.hparams.out_channels), dtype=torch.long, device=device)
            active_indices[:, 0] = 1
            for idx_active in range(len(self.hparams.label_indices_downstream_active)):
                active_indices[:, self.hparams.label_indices_downstream_active[idx_active]] = 1
            self.label_indices_active = active_indices

    def set_frozen_indices(self,
                           device: torch.device):
        # Frozen categories - Does not impact the selection of instructions
        if not self.hparams.downstream:
            self.label_indices_frozen = torch.zeros((self.hparams.out_channels,), dtype=torch.long, device=device)  # There are currently no frozen categories during the "normal" training routine
        else:
            frozen_indices = torch.zeros((self.hparams.out_channels, ), dtype=torch.long, device=device)
            for idx_frozen in range(len(self.hparams.label_indices_downstream_frozen)):
                frozen_indices[self.hparams.label_indices_downstream_frozen[idx_frozen]] = 1
            self.label_indices_frozen = frozen_indices

    def scatter_predictions(self, preds: torch.Tensor):
        """ Scatter preds for each batch-element (with possibly different indices) to respective label indices """
        preds_scattered = torch.zeros((preds.shape[0], self.hparams.out_channels, *preds.shape[2:]), dtype=preds.dtype, device=preds.device)
        for idx_batch in range(preds.shape[0]):
            preds_indices = einops.rearrange(torch.nonzero(self.label_indices_active[idx_batch, :], as_tuple=False).squeeze(), 'c -> c () () ()').expand(-1, *preds.shape[2:])
            preds_scattered[idx_batch, ...].scatter_add_(dim=0, index=preds_indices, src=preds[idx_batch, ...])  # optionally: - preds[idx_batch, ...].detach().min() - min to ensure to avoid going lower than initialization (0)?
        return preds_scattered

    def gather_labels(self, targets_one_hot: torch.Tensor):
        """ Gather targets for each batch-element (with possibly different indices) to respective prediction indices """
        targets_one_hot_gathered = list()
        targets_one_hot_active = list()
        for idx_batch in range(targets_one_hot.shape[0]):
            # Select active labels (can vary for each batch element)
            targets_indices = einops.rearrange(torch.nonzero(self.label_indices_active[idx_batch, :], as_tuple=False).squeeze(), 'c -> c () () ()').expand(-1, *targets_one_hot.shape[2:])
            targets_element = torch.gather(targets_one_hot[idx_batch, ...], dim=0, index=targets_indices)

            # Add inactive labels to background
            targets_element_inactive = targets_one_hot[idx_batch].clone()
            targets_element_inactive[self.label_indices_active[idx_batch, :] == 1, ...] = 0  # Mask active elements
            targets_element[0, ...] += targets_element_inactive.sum(dim=0)  # Add inactive elements to background
            targets_one_hot_gathered.append(targets_element)

            # New target for metric comparison
            targets_element_active = targets_one_hot[idx_batch].clone()
            targets_element_active[0, ...] += targets_element_inactive.sum(dim=0)  # Add inactive elements to background
            targets_element_active[self.label_indices_active[idx_batch, :] == 0, ...] = 0  # Mask inactive elements
            targets_one_hot_active.append(targets_element_active)

        targets_one_hot_gathered = torch.stack(targets_one_hot_gathered, dim=0)
        targets_gathered = torch.argmax(targets_one_hot_gathered, dim=1)
        targets_one_hot_active = torch.stack(targets_one_hot_active, dim=0)

        return targets_one_hot_gathered, targets_gathered, targets_one_hot_active

    def gather_non_frozen_predictions(self, preds_scattered: torch.Tensor):
        """
         Gather preds so only non-frozen & active categories are considered as foreground. Thus, each frozen category (active or not) is aggregated into background.
         Operating on preds_scattered for streamlined gather process.
        """
        n_batch = preds_scattered.shape[0]

        # Scatter inactive / frozen in background, active & non-frozen into consecutive foreground dims
        label_indices_all = torch.arange(self.hparams.out_channels, dtype=torch.long, device=preds_scattered.device)
        label_indices_frozen_ = self.label_indices_frozen.clone()
        label_indices_frozen_[0] = 0  # For predictions / targets a background is needed, so it is always considered "non-frozen" for prediction / target aggregation (regardless if its truly frozen or not)
        labels_active_and_nonfrozen_mask = torch.logical_and(self.label_indices_active.bool(), ~label_indices_frozen_.unsqueeze(dim=0).expand(n_batch, -1).bool()).long()
        predictions_non_frozen = torch.zeros((preds_scattered.shape[0],
                                              torch.count_nonzero(labels_active_and_nonfrozen_mask[0, ...]).item(),  # All batch elements should have the same amount of active elements (to keep it in a tensor)
                                              *preds_scattered.shape[2:]), dtype=preds_scattered.dtype, device=preds_scattered.device)
        for idx_batch in range(n_batch):
            label_indices_active_and_nonfrozen = labels_active_and_nonfrozen_mask[idx_batch, ...] * label_indices_all  # all inactive / frozen ones are zero
            _, inverse_indices = torch.unique(label_indices_active_and_nonfrozen, sorted=True, return_inverse=True)
            preds_indices = einops.rearrange(inverse_indices, 'c -> c () () ()').expand(-1, *preds_scattered.shape[2:])
            predictions_non_frozen[idx_batch, ...].scatter_add_(dim=0, index=preds_indices, src=preds_scattered[idx_batch, ...])

        return predictions_non_frozen

    def gather_non_frozen_labels(self, targets_one_hot: torch.Tensor):
        """ Gather targets so only non-frozen & active categories are considered as foreground. Thus, each frozen category (active or not) is aggregated into background """
        n_batch = targets_one_hot.shape[0]

        # Scatter inactive / frozen in background, active & non-frozen into consecutive foreground dims
        label_indices_all = torch.arange(self.hparams.out_channels, dtype=torch.long, device=targets_one_hot.device)
        label_indices_frozen_ = self.label_indices_frozen.clone()
        label_indices_frozen_[0] = 0  # For predictions / targets a background is needed, so it is always considered "non-frozen" for prediction / target aggregation (regardless if its truly frozen or not)
        labels_active_and_nonfrozen_mask = torch.logical_and(self.label_indices_active.bool(), ~label_indices_frozen_.unsqueeze(dim=0).expand(n_batch, -1).bool()).long()
        targets_one_hot_non_frozen = torch.zeros((targets_one_hot.shape[0],
                                                  torch.count_nonzero(labels_active_and_nonfrozen_mask[0, ...]).item(),  # All batch elements should have the same amount of active elements (to keep it in a tensor)
                                                  *targets_one_hot.shape[2:]), dtype=targets_one_hot.dtype, device=targets_one_hot.device)
        for idx_batch in range(n_batch):
            label_indices_active_and_nonfrozen = labels_active_and_nonfrozen_mask[idx_batch, ...] * label_indices_all  # all inactive / frozen ones are zero
            _, inverse_indices = torch.unique(label_indices_active_and_nonfrozen, sorted=True, return_inverse=True)
            preds_indices = einops.rearrange(inverse_indices, 'c -> c () () ()').expand(-1, *targets_one_hot.shape[2:])
            targets_one_hot_non_frozen[idx_batch, ...].scatter_add_(dim=0, index=preds_indices, src=targets_one_hot[idx_batch, ...])
        targets_non_frozen = torch.argmax(targets_one_hot_non_frozen, dim=1)

        return targets_one_hot_non_frozen, targets_non_frozen

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"].copy()
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                # Adjust parameters with size mismatch
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")
                    checkpoint["state_dict"][k] = model_state_dict[k]
                    is_changed = True
            else:
                # Remove parameters not in the actual model
                warnings.warn(f"Dropping parameter: {k}")
                del checkpoint["state_dict"][k]
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.s3_bucket and self.hparams.online_on:
            print(f'\rUploading checkpoint to {self.hparams.ckpt_dir} ...')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--learning_rate_downstream', default=5e-4, type=float)
        parser.add_argument('--learning_rate_instructions', default=1e-3, type=float)
        parser.add_argument('--learning_rate_instructions_downstream', default=5e-3, type=float)
        parser.add_argument('--weight_decay', default=1e-2, type=float)
        parser.add_argument('--weight_decay_downstream', default=0, type=float)
        parser.add_argument('--with_scheduler_downstream', default=True, type=bool)
        parser.add_argument('--sliding_window_overlap', default=0.5, type=float)

        # Segmentation loss
        parser.add_argument('--loss_weight_segmentation', default=1e-0, type=float)
        parser.add_argument('--loss_weight_segmentation_gamma', default=4.0, type=float)
        parser.add_argument('--loss_weight_segmentation_alpha_background', default=1.0, type=float)
        parser.add_argument('--loss_weight_segmentation_alpha_background_downstream', default=1.0, type=float)  # May use higher value than during training, to avoid fast collapse of background into foreground.
        parser.add_argument('--loss_weight_segmentation_alpha_foreground', default=1.0, type=float)

        # Contrastive loss
        parser.add_argument('--contrastive_pairs', default=True, type=bool)
        parser.add_argument('--loss_weight_sim_paired', default=0., type=float)
        parser.add_argument('--loss_weight_sim_protos', default=1e-2, type=float)
        parser.add_argument('--loss_weight_sim_protos_downstream', default=0, type=float)
        parser.add_argument('--loss_weight_sim_closest', default=0., type=float)
        parser.add_argument('--loss_weight_dissim_closest', default=0., type=float)
        parser.add_argument('--k_means_iterations', default=3, type=int)
        parser.add_argument('--reduction_factor', default=2., type=float)
        parser.add_argument('--reduction_factor_protos', default=8., type=float)
        parser.add_argument('--fwhm_student_teacher', default=128., type=float)
        parser.add_argument('--fwhm_teacher_protos', default=128., type=float)
        parser.add_argument('--use_weighting_protos', default=True, type=bool)
        parser.add_argument('--use_weighting_teacher', default=False, type=bool)

        # Instruction swap
        parser.add_argument('--label_indices_min_active', default=1, type=int)  # Minimal amount of active FOREGROUND categories during training
        parser.add_argument('--label_indices_max_active', default=1, type=int)  # Maximal amount of active FOREGROUND categories during training
        parser.add_argument('--label_indices_base', default=[1], nargs='*', type=int)  # Without background (which is expected to be 0)
        parser.add_argument('--label_indices_downstream_active', default=[4], nargs='*', type=int)  # Active foreground classes (frozen / non-frozen) in the downstream task (without background which is expected to be 0)
        parser.add_argument('--label_indices_downstream_frozen', default=[], nargs='*', type=int)  # Active but frozen foreground (AND background) classes (which will be aggregated into background during "downstream" training so no "old" annotations are required in the case of class extension).
        parser.add_argument('--downstream', action='store_true')  # Train solely categories in label_indices_downstream_active
        parser.add_argument('--selective_freezing', action='store_true')  # Freeze the network except for active instructions not in label_indices_downstream_frozen
        parser.add_argument('--freeze_body', default=True, type=bool)  # Freeze body if selective_freezing is active
        parser.add_argument('--freeze_norm', default=False, type=bool)  # Freeze instruction norm if selective_freezing is active
        parser.add_argument('--freeze_bias_scores', default=False, type=bool)  # Freeze instruction bias scores if selective_freezing is active
        parser.add_argument('--freeze_inactive', default=True, type=bool)  # Freeze inactive instructions regardless of whether they are frozen or not
        parser.add_argument('--separate_background', default=True, type=bool)  # Each foreground class has its own background (for binary prediction case)

        return parser
