from argparse import ArgumentParser
import pathlib as plb
import pytorch_lightning as pl
import os


def setup_checkpointing(hparams, save_top_k: int = 3):
    # Note: requires hparams attributes that might only be set in setup_loggers()

    # configure checkpoint directory
    if hparams.s3_bucket.strip(' ') and hparams.online_on:
        # bucket directory
        dir_ckpt = os.path.join(hparams.s3_bucket, 'checkpoints', hparams.run_name)
    else:
        # offline directory
        dir_ckpt = str(plb.Path(hparams.default_root_dir) / 'checkpoints' / hparams.run_name)
    hparams.__dict__['ckpt_dir'] = dir_ckpt

    save_last = True if hparams.ckpt_save_last else False
    if hparams.loss_weight_segmentation > 0.:
        monitor_ = 'val_dice_mean'
        mode_ = 'max'
        filename_ = f'ckpt_{{epoch:03d}}_{{val_dice_mean:.4f}}'
    else:
        monitor_ = 'train_loss_mean'
        mode_ = 'min'
        filename_ = f'ckpt_{{epoch:03d}}_{{train_loss_mean:.4f}}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor_,
                                                       mode=mode_,
                                                       save_last=save_last,
                                                       save_top_k=save_top_k,
                                                       dirpath=dir_ckpt,
                                                       filename=filename_,
                                                       verbose=True)

    return checkpoint_callback


def add_callback_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--ckpt_save_last', action='store_true')
    return parser
