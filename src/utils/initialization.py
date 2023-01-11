from typing import Type
import pytorch_lightning as pl
import pathlib as plb
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torch
from src.utils import callbacks, logging_custom
import warnings


def setup_training(hparams, cls_dm: Type[pl.LightningDataModule], cls_model: Type[pl.LightningModule], path_root):

    # Load and adjust hparams
    if hparams.no_overwrite:
        if hparams.ckpt is None:
            warnings.warn(f'No checkpoint is available, despite flag --no_overwrite being active.')
        else:
            hparams_ckpt = pl_load(hparams.ckpt, map_location=lambda storage, loc: storage)['hyper_parameters']

            # Overwrite some important (local) values
            hparams_ckpt['no_overwrite'] = hparams.no_overwrite
            hparams_ckpt['cold_start'] = hparams.cold_start
            hparams_ckpt['gpus'] = hparams.gpus
            hparams_ckpt['accelerator'] = hparams.accelerator
            hparams_ckpt['plugins'] = hparams.plugins
            hparams_ckpt['online_on'] = hparams.online_on
            hparams_ckpt['checkpoint_callback'] = hparams.checkpoint_callback  # Lightning has become picky ...
            hparams_ckpt['terminate_on_nan'] = False  # Lightning has become picky ...
            hparams_ckpt['accumulate_grad_batches'] = 1  # Lightning has become picky ...
            hparams_ckpt['ckpt'] = hparams.ckpt
            hparams_ckpt['tags'] = hparams.tags if hasattr(hparams, 'tags') else hparams_ckpt['tags']
            hparams_ckpt['mode'] = hparams.mode
            hparams_ckpt['architecture'] = hparams.architecture
            hparams_ckpt['downstream'] = hparams.downstream
            hparams_ckpt['separate_background'] = hparams.separate_background
            hparams_ckpt['num_annotated'] = hparams.num_annotated
            hparams_ckpt['loss_weight_segmentation'] = hparams.loss_weight_segmentation
            hparams_ckpt['loss_weight_segmentation_gamma'] = hparams.loss_weight_segmentation_gamma
            hparams_ckpt['loss_weight_segmentation_alpha_background'] = hparams.loss_weight_segmentation_alpha_background
            hparams_ckpt['loss_weight_segmentation_alpha_background_downstream'] = hparams.loss_weight_segmentation_alpha_background_downstream
            hparams_ckpt['loss_weight_segmentation_alpha_foreground'] = hparams.loss_weight_segmentation_alpha_foreground
            hparams_ckpt['additive_alpha'] = hparams.additive_alpha
            hparams_ckpt['loss_weight_sim_protos_downstream'] = hparams.loss_weight_sim_protos_downstream
            hparams_ckpt['learning_rate'] = hparams.learning_rate
            hparams_ckpt['learning_rate_downstream'] = hparams.learning_rate_downstream
            hparams_ckpt['learning_rate_instructions'] = hparams.learning_rate_instructions
            hparams_ckpt['learning_rate_instructions_downstream'] = hparams.learning_rate_instructions_downstream
            hparams_ckpt['weight_decay'] = hparams.weight_decay
            hparams_ckpt['weight_decay_downstream'] = hparams.weight_decay_downstream
            hparams_ckpt['with_scheduler_downstream'] = hparams.with_scheduler_downstream
            hparams_ckpt['batch_size'] = hparams.batch_size
            hparams_ckpt['max_steps'] = hparams.max_steps
            hparams_ckpt['max_epochs'] = hparams.max_epochs
            hparams_ckpt['num_samples_epoch'] = hparams.num_samples_epoch
            hparams_ckpt['flush_logs_every_n_steps'] = hparams.flush_logs_every_n_steps

            # Replace keys not available in all cases
            for key_ in ['adaptation_variant', 'prompting_variant', 'selective_freezing',
                         'freeze_body', 'freeze_norm', 'freeze_bias_scores', 'freeze_inactive', 'fixed_output',
                         'tokens_per_instruction', 'mean_aggregation', 'top_k_selection',
                         'soft_selection_sigma', 'noninstructed_attention', 'no_bias_instructions', 'instructions_use_norm', 'instructions_elementwise_affine',
                         'label_indices_base', 'label_indices_downstream_active', 'label_indices_max_active']:
                if hasattr(hparams, key_):
                    hparams_ckpt[key_] = getattr(hparams, key_)

            # Misc
            hparams_ckpt['tmp_dir'] = hparams.tmp_dir
            hparams_ckpt['export_dir'] = str(plb.Path(hparams.export_dir) / plb.Path(hparams.ckpt).parent.name) if hparams.export_dir else str(plb.Path(hparams.ckpt).parent / 'predictions')
            hparams_ckpt['dir_images'] = hparams.dir_images
            hparams_ckpt['dir_masks'] = hparams.dir_masks
            hparams_ckpt['default_root_dir'] = str(path_root / 'logs' / 'lightning')
            hparams_ckpt['gpus'] = hparams.gpus
            hparams_ckpt['accelerator'] = hparams.accelerator
            hparams_ckpt['plugins'] = hparams.plugins

            # Logging
            hparams_ckpt['online_on'] = hparams.online_on
            hparams_ckpt['plot_interval_train'] = hparams.plot_interval_train
            hparams_ckpt['plot_interval_val'] = hparams.plot_interval_val
            hparams_ckpt['plot_interval_test'] = hparams.plot_interval_test

            # Remove unwanted keys
            del hparams_ckpt['run_name']

            for k_, v_ in hparams_ckpt.items():
                if k_ in hparams.__dict__.keys():
                    print(f'Overwriting {k_} with {v_} from passed args / ckpt instead of {hparams.__dict__[k_]}')
                else:
                    print(f'Setting {k_} to {v_} from passed args / ckpt.')
            hparams.__dict__.update(hparams_ckpt)

    # Logging
    loggers = logging_custom.setup_loggers(hparams, path_root=path_root)
    ckpt_callback = callbacks.setup_checkpointing(hparams)

    # Setup
    dm = cls_dm(hparams)
    dm.prepare_data()
    dm.setup()  # Currently raises a deprecation warning.

    if hparams.ckpt is not None and hparams.cold_start:
        model = cls_model.load_from_checkpoint(hparams.ckpt,
                                               **hparams.__dict__,
                                               strict=False if hparams.downstream else True)  # overwrite params
        resume_from_checkpoint = None
    else:
        model = cls_model(hparams)
        resume_from_checkpoint = hparams.ckpt

    # Train model
    check_val_every_n_epoch_ = hparams.check_val_every_n_epoch_ if hasattr(hparams, 'check_val_every_n_epoch_') and hparams.check_val_every_n_epoch_ is not None else hparams.check_val_every_n_epoch  # set via custom argparse parameter
    check_val_every_n_epoch_ = check_val_every_n_epoch_ if not hparams.downstream else hparams.check_val_every_n_epoch_downstream
    trainer = Trainer.from_argparse_args(hparams,
                                         gpus=hparams.gpus,  # always set gpus via run config / cmd line arguments
                                         gradient_clip_val=1.,
                                         gradient_clip_algorithm='value',
                                         check_val_every_n_epoch=check_val_every_n_epoch_,
                                         resume_from_checkpoint=resume_from_checkpoint,
                                         callbacks=[ckpt_callback],
                                         logger=loggers,
                                         precision=16 if 'wip' in hparams.architecture else 32)  # limit_train_batches=10, num_sanity_val_steps=0

    model.summarize(max_depth=-1)  # avoid calling this during trainer creation due to potential logging buffer congestion

    return dm, model, trainer


def setup_testing(hparams, cls_dm: Type[pl.LightningDataModule], cls_model: Type[pl.LightningModule], path_root):

    # Adjust hparams
    assert hparams.ckpt is not None
    hparams_ckpt = pl_load(hparams.ckpt, map_location=lambda storage, loc: storage)['hyper_parameters']

    # Overwrite some important (local) values
    hparams_ckpt['no_overwrite'] = hparams.no_overwrite
    hparams_ckpt['cold_start'] = hparams.cold_start
    hparams_ckpt['gpus'] = hparams.gpus
    hparams_ckpt['accelerator'] = hparams.accelerator
    hparams_ckpt['plugins'] = hparams.plugins
    hparams_ckpt['online_on'] = False
    hparams_ckpt['checkpoint_callback'] = hparams.checkpoint_callback  # Lightning has become picky ...
    hparams_ckpt['terminate_on_nan'] = False  # Lightning has become picky ...
    hparams_ckpt['accumulate_grad_batches'] = 1  # Lightning has become picky ...
    hparams_ckpt['ckpt'] = hparams.ckpt
    hparams_ckpt['mode'] = hparams.mode
    hparams_ckpt['flush_logs_every_n_steps'] = hparams.flush_logs_every_n_steps

    # Remove outdated keys
    if 'freeze_bias' in hparams_ckpt:
        del hparams_ckpt['freeze_bias']
    if 'learning_rate_cnn' in hparams_ckpt:
        del hparams_ckpt['learning_rate_cnn']

    # Misc
    hparams_ckpt['tmp_dir'] = hparams.tmp_dir
    hparams_ckpt['export_dir'] = str(plb.Path(hparams.export_dir) / plb.Path(hparams.ckpt).parent.name) if hparams.export_dir else str(plb.Path(hparams.ckpt).parent / 'predictions')
    hparams_ckpt['dir_images'] = hparams.dir_images
    hparams_ckpt['dir_masks'] = hparams.dir_masks
    hparams_ckpt['default_root_dir'] = str(path_root / 'logs' / 'lightning')
    hparams_ckpt['gpus'] = hparams.gpus
    hparams_ckpt['accelerator'] = hparams.accelerator
    hparams_ckpt['plugins'] = hparams.plugins

    # Logging
    hparams_ckpt['online_on'] = hparams.online_on
    hparams_ckpt['plot_interval_train'] = hparams.plot_interval_train
    hparams_ckpt['plot_interval_val'] = hparams.plot_interval_val
    hparams_ckpt['plot_interval_test'] = hparams.plot_interval_test
    hparams.__dict__.update(hparams_ckpt)

    # Logging
    loggers = logging_custom.setup_loggers(hparams, path_root=path_root)

    # Setup
    dm = cls_dm(hparams)
    dm.prepare_data()
    dm.setup()  # Currently raises a deprecation warning.

    model = cls_model.load_from_checkpoint(hparams.ckpt,
                                           **hparams.__dict__)  # overwrite parameters
    trainer = Trainer.from_argparse_args(hparams,
                                         gradient_clip_val=1.,
                                         gradient_clip_algorithm='value',
                                         gpus=hparams.gpus,
                                         logger=loggers,
                                         resume_from_checkpoint=None)

    return dm, model, trainer
