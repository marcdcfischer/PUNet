from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Any, Dict, Union, List, Tuple
import pathlib as plb
from src.modules.architectures.momentum_model_simple import MomentumModelSimple
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


# This module is used for the baseline architectures (available through monai)
class InstructionModelSimple(pl.LightningModule):
    def __init__(self, conf: Union[Dict, Namespace], **kwargs):
        super().__init__()
        self.save_hyperparameters(conf)

        print(f'Establishing architecture with parameters: \n {self.hparams}')

        # Architecture
        self.architecture = MomentumModelSimple(conf=self.hparams)

        # Losses
        # Segmentation losses
        self.loss_seg = focal.FocalLoss(self.hparams.out_channels,
                                        loss_weight=self.hparams.loss_weight_segmentation,
                                        gamma=self.hparams.loss_weight_segmentation_gamma,
                                        alpha_background=self.hparams.loss_weight_segmentation_alpha_background if not self.hparams.downstream else self.hparams.loss_weight_segmentation_alpha_background_downstream,
                                        alpha_foreground=self.hparams.loss_weight_segmentation_alpha_foreground,
                                        additive_alpha=self.hparams.additive_alpha)

        # Contrastive losses
        # Account for half res of wip architecture
        reduction_factor_ = self.hparams.reduction_factor if self.hparams.architecture == "wip_simple" else self.hparams.reduction_factor * 2
        reduction_factor_protos_ = self.hparams.reduction_factor_protos if self.hparams.architecture == "wip_simple" else self.hparams.reduction_factor_protos * 2
        print(f'Using (adjusted) reduction factor: {reduction_factor_} and reduction_factor_protos: {reduction_factor_protos_}.')
        self.loss_cluster_pairs = ContrastiveProtosTeacherLoss(
            reduction_factor=reduction_factor_,
            reduction_factor_protos=reduction_factor_protos_,
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
                teacher_prediction: bool = True,
                second_student_prediction: bool = True):

        x_teacher = x_teacher if teacher_prediction else None
        x = x if second_student_prediction else x[:1]
        dict_out_students, dict_out_teacher = self.architecture(x, x_teacher)

        return dict_out_students, dict_out_teacher

    def forward_prediction(self, x: torch.Tensor):  # takes an image and predicts one (as required for monai sliding window inference)

        dict_out_students, _ = self.architecture([x], None)

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

        # Batch preparations
        # Aux content for one of the cateogires (student large) only
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

        # Momentum update
        self.architecture.update_teacher()

        # prediction
        dict_out_students, dict_out_teacher = self(x_students,
                                                   x_teacher,
                                                   teacher_prediction=self.loss_cluster_pairs.loss_weight > 0.,
                                                   second_student_prediction=self.loss_cluster_pairs.loss_weight > 0.)

        # Loss calculation
        loss_dict, plots_dict = dict(), dict()
        # Segmentation
        if self.hparams.loss_weight_segmentation > 0.:
            for idx_student in range(len(dict_out_students)):
                loss_dict.update(self.loss_seg(dict_out_students[idx_student]['dense']['embedded_latents'][aux_annotated, ...], y_students[idx_student][aux_annotated, ...], tag=f'seg_s{idx_student}'))
        else:
            loss_dict['seg'] = torch.tensor(0., device=self.device, requires_grad=False)

        # Self-supervised contrastive
        if self.hparams.contrastive_pairs:
            if self.loss_cluster_pairs.loss_weight > 0.:
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
                predictions_one_hot_ = F.one_hot(dict_out_students[idx_student]['dense']['embedded_latents'].argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
                self.score_seg_train(y_pred=predictions_one_hot_,
                                     y=y_students_one_hot[idx_student])
                if any(aux_annotated):
                    self.score_seg_train_annotated(y_pred=predictions_one_hot_[aux_annotated, ...],
                                                   y=y_students_one_hot[idx_student][aux_annotated, ...])
                if any(~aux_annotated):
                    self.score_seg_train_non_annotated(y_pred=predictions_one_hot_[~aux_annotated, ...],
                                                       y=y_students_one_hot[idx_student][~aux_annotated, ...])

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
                        png_paths.extend(similarities_student_teacher.visualize_similarities_student_teacher({k_: v_ for k_, v_ in plots_dict.items() if f'{idx_student}' in k_}, x_students[idx_student], x_teacher,
                                                                                                             prefix=f'{self.hparams.run_name}_train_b{batch_idx}_s{idx_student}_ep{str(self.current_epoch).zfill(3)}', path_plots=self.hparams.default_root_dir))
                        for identifier_, dict_, x_, y_one_hot_, in zip(['student', 'teacher'], [dict_out_students[idx_student], dict_out_teacher], [x_students[idx_student], x_teacher], [y_students_one_hot[idx_student], y_teacher_one_hot]):
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

        # Sliding Window Inference - by teacher
        volume_prediction = minferers.sliding_window_inference(inputs=x,
                                                               roi_size=self.hparams.patch_size_students[0],
                                                               sw_batch_size=self.hparams.batch_size,
                                                               predictor=partial(self.forward_prediction),
                                                               overlap=self.hparams.sliding_window_overlap)

        # Loss calculation
        loss_dict, plots_dict = dict(), dict()
        # Segmentation
        if any(aux_annotated) and self.hparams.loss_weight_segmentation > 0.:
            loss_dict.update(self.loss_seg(volume_prediction[aux_annotated, ...], y[aux_annotated, ...]))
        else:
            loss_dict['seg'] = torch.tensor(0., device=self.device, requires_grad=False)
        # Self-supervised contrastive not available in the current setup, since patched embeddings would be required
        loss_dict['all'] = sum(loss_dict.values())
        loss_dict_all.append(loss_dict)

        with torch.no_grad():
            # metrics
            predictions_one_hot = F.one_hot(volume_prediction.argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
            self.score_seg_val(y_pred=predictions_one_hot,
                               y=y_one_hot)

            # plotting
            if self.hparams.plot and (self.current_epoch + 1) % self.hparams.plot_interval_val == 0 and batch_idx % 10 == 0 and batch_idx <= 20:
                if not dist.is_initialized() or dist.get_rank() < 1:
                    png_paths = list()
                    png_paths.extend(image_grid.plot_grid_middle(x.detach().cpu(), y_one_hot.detach().cpu(),
                                                                 torch.softmax(volume_prediction.detach().cpu(), dim=1),
                                                                 scribbles.detach().cpu() if scribbles is not None else None,
                                                                 indices_elements=[idx_ * 2 for idx_ in range(max(min(x.shape[0] // 2, 2), 1))],
                                                                 prefix=f'{self.hparams.run_name}_val_b{batch_idx}_s0_c-all_ep{str(self.current_epoch).zfill(3)}', path_plots=self.hparams.default_root_dir))
                    if self.hparams.online_on:
                        [self.logger[1].experiment.log({str(plb.Path(png_).stem[:-6]): wandb.Image(png_)}) for png_ in png_paths]

        loss_dict_all = {k_: sum(dict_[k_] for dict_ in loss_dict_all) / len(loss_dict_all) for k_ in loss_dict_all[0].keys()}

        with torch.no_grad():
            # logging
            self.log(f'val_epoch', self.current_epoch, sync_dist=True)  # log for ckpt selection
            for key_, value_ in loss_dict_all.items():
                self.log(f'val_loss_{key_}', value_, sync_dist=True)  # logs per epoch

        return {'loss': loss_dict_all['all']}

    def validation_epoch_end(self, outputs) -> None:
        self.log(f'val_loss_mean', torch.mean(torch.tensor([x_['loss'] for x_ in outputs], dtype=torch.float)), sync_dist=True)
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

        # Sliding Window Inference - by teacher
        volume_prediction = minferers.sliding_window_inference(inputs=x,
                                                               roi_size=self.hparams.patch_size_students[0],
                                                               sw_batch_size=self.hparams.batch_size,
                                                               predictor=partial(self.forward_prediction),
                                                               overlap=self.hparams.sliding_window_overlap)

        # Loss calculation
        loss_dict, plots_dict = dict(), dict()
        # Segmentation
        if any(aux_annotated) and self.hparams.loss_weight_segmentation > 0.:
            loss_dict.update(self.loss_seg(volume_prediction[aux_annotated, ...], y[aux_annotated, ...]))
        else:
            loss_dict['seg'] = torch.tensor(0., device=self.device, requires_grad=False)
        # Self-supervised contrastive not available in the current setup, since patched embeddings would be required
        loss_dict['all'] = sum(loss_dict.values())
        loss_dict_all.append(loss_dict)

        with torch.no_grad():
            # metrics
            predictions_one_hot = F.one_hot(volume_prediction.argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
            self.score_seg_test(y_pred=predictions_one_hot,
                                y=y_one_hot)

            # save predictions
            test_viz = False
            if test_viz:
                import matplotlib
                matplotlib.use('tkagg')
                argmaxed = torch.argmax(volume_prediction[0, ...], dim=0)
                viewer = nib.viewers.OrthoSlicer3D(np.array((argmaxed / argmaxed.max()).detach().cpu()))
                viewer.clim = [0.0, 1.0]
                viewer.show()
            post_transform = generate_test_post_transforms(output_dir=self.hparams.export_dir,
                                                           output_postfix=f'pred_cat-all',
                                                           transform_test=self.trainer.datamodule.transform_test,
                                                           n_classes=None)
            batch['pred'] = volume_prediction
            [post_transform(x_) for x_ in decollate_batch(batch)]

            # plotting
            if self.hparams.plot and (self.current_epoch + 1) % self.hparams.plot_interval_test == 0 and batch_idx % 10 == 0 and batch_idx <= 20:
                if not dist.is_initialized() or dist.get_rank() < 1:
                    png_paths = list()
                    png_paths.extend(image_grid.plot_grid_middle(x.detach().cpu(), y_one_hot.detach().cpu(),
                                                                 torch.softmax(volume_prediction.detach().cpu(), dim=1),
                                                                 scribbles.detach().cpu() if scribbles is not None else None,
                                                                 indices_elements=[idx_ * 2 for idx_ in range(max(min(x.shape[0] // 2, 2), 1))],
                                                                 prefix=f'{self.hparams.run_name}_test_b{batch_idx}_s0_c-all_ep{str(self.current_epoch).zfill(3)}', path_plots=self.hparams.default_root_dir))
                    if self.hparams.online_on:
                        [self.logger[1].experiment.log({str(plb.Path(png_).stem[:-6]): wandb.Image(png_)}) for png_ in png_paths]

        loss_dict_all = {k_: sum(dict_[k_] for dict_ in loss_dict_all) / len(loss_dict_all) for k_ in loss_dict_all[0].keys()}

        with torch.no_grad():
            # logging
            self.log(f'test_epoch', self.current_epoch, sync_dist=True)  # log for ckpt selection
            for key_, value_ in loss_dict_all.items():
                self.log(f'test_loss_{key_}', value_, sync_dist=True)  # logs per epoch

        return {'loss': loss_dict_all['all']}

    def test_epoch_end(self, outputs) -> None:
        self.log(f'test_loss_mean', torch.mean(torch.tensor([x_['loss'] for x_ in outputs], dtype=torch.float)), sync_dist=True)
        score_seg = self.score_seg_test.aggregate()
        self.score_seg_test.reset()
        self.log_dict({f'test_dice_c{str(idx_c).zfill(2)}': score_seg[idx_c] for idx_c in range(len(score_seg))}, sync_dist=True)
        self.log(f'test_dice_mean', torch.mean(score_seg[1:]), sync_dist=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        pass

    def configure_optimizers(self):
        if self.hparams.architecture == "wip_simple":
            optimizer = AdamW([{'params': [param_ for name_, param_ in self.architecture.network_student.get_named_body_parameters()]},
                               {'params': [param_ for name_, param_ in self.architecture.network_student.get_named_instruction_parameters()],
                                'lr': self.hparams.learning_rate_instructions if not self.hparams.downstream else self.hparams.learning_rate_instructions_downstream}],
                              lr=self.hparams.learning_rate if not self.hparams.downstream else self.hparams.learning_rate_downstream,
                              weight_decay=self.hparams.weight_decay if not self.hparams.downstream else self.hparams.weight_decay_downstream)
        else:
            optimizer = AdamW(self.parameters(),
                              lr=self.hparams.learning_rate if not self.hparams.downstream else self.hparams.learning_rate_downstream,
                              weight_decay=self.hparams.weight_decay if not self.hparams.downstream else self.hparams.weight_decay_downstream)

        if self.hparams.downstream and self.hparams.with_scheduler_downstream:
            assert self.hparams.max_epochs is not None
            print(f'Using one cycle lr scheduler for {self.hparams.max_epochs} epochs and {1} steps per epoch.')
            scheduler = lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=[self.hparams.learning_rate_downstream, self.hparams.learning_rate_instructions_downstream] if self.hparams.architecture == "wip_simple" else self.hparams.learning_rate_downstream,
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

        # Misc
        parser.add_argument('--label_indices_max_active', default=-1, type=int)  # Should stay negative for simple case.
        parser.add_argument('--downstream', action='store_true')
        parser.add_argument('--label_indices_base', default=[], nargs='*', type=int)  # Does not have any effect atm.
        parser.add_argument('--label_indices_downstream_active', default=[], nargs='*', type=int)  # Does not have any effect atm.
        parser.add_argument('--separate_background', default=False, type=bool)  # Should stay False for simple case.

        return parser
