import argparse
from argparse import ArgumentParser
import pytorch_lightning as pl
from typing import Type
import pathlib as plb
from src.utils import aws
from src.data.loader_monai import BasicDataModule
from src.modules.instruction_model import InstructionModel as Model
from src.modules.instruction_model_simple import InstructionModelSimple as ModelSimple
from src.modules.architectures.momentum_model import MomentumModel as Architecture
from src.modules.architectures.momentum_model_simple import MomentumModelSimple as ArchitectureSimple
from src.utils import initialization, callbacks


def main(hparams: argparse.Namespace, cls_dm: Type[pl.LightningDataModule], cls_model: Type[pl.LightningModule]):
    path_root = plb.Path(__file__).resolve().parent.parent
    if hparams.mode == 'fit':
        pl.seed_everything(1234, workers=True)
        dm, model, trainer = initialization.setup_training(hparams, cls_dm=cls_dm, cls_model=cls_model, path_root=path_root)
        trainer.fit(model, datamodule=dm)
    elif hparams.mode == 'validate':
        pl.seed_everything(2345, workers=True)
        dm, model, trainer = initialization.setup_training(hparams, cls_dm=cls_dm, cls_model=cls_model, path_root=path_root)
        trainer.validate(model, datamodule=dm)
    elif hparams.mode == 'test':
        pl.seed_everything(3456, workers=True)
        dm, model, trainer = initialization.setup_testing(hparams, cls_dm=cls_dm, cls_model=cls_model, path_root=path_root)
        trainer.test(model, datamodule=dm)
    elif hparams.mode == 'predict':
        pl.seed_everything(4567, workers=True)
        dm, model, trainer = initialization.setup_testing(hparams, cls_dm=cls_dm, cls_model=cls_model, path_root=path_root)
        trainer.predict(model, datamodule=dm)
    else:
        raise ValueError(f'The mode {hparams.mode} is not available.')


# Exemplary execution
# See shell/common/cfgs.sh for variants
# Training: --gpus 1 --batch_size 8 --architecture wip --dataset tcia_btcv
# Downstream: --gpus 1 --batch_size 8 --architecture wip --dataset tcia_btcv --adaptation_variant prompting --no_overwrite --cold_start --selective_freezing --downstream --label_indices_base 1 --label_indices_downstream_active 2 --max_epochs 100
# Test: --gpus 1 --mode test --architecture wip --dataset tcia_btcv --no_overwrite --cold_start
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--s3_bucket', nargs='?', const='', default='', type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--ckpt_run_name', default=None, type=str)  # e.g. med_rep_learning_XXX
    parser.add_argument('--cold_start', action='store_true')  # Ignores the optimizer state dict
    parser.add_argument('--no_overwrite', action='store_true')  # Keep (most) hparameters of saved checkpoint
    parser.add_argument('--plot', default=True, type=bool)
    parser.add_argument('--online_logger', default='wandb', type=str)
    parser.add_argument('--online_entity', default='my_entity', type=str)  # EDIT ME (if needed)
    parser.add_argument('--online_project', default='my_project', type=str)  # EDIT ME (if needed)
    parser.add_argument('--online_on', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str)
    parser.add_argument('--tmp_dir', default='/tmp', type=str)
    parser.add_argument('--export_dir', default='/mnt/SSD_SATA_03/data_med/prompting/output', type=str)  # EDIT ME
    parser.add_argument('--mode', default='fit', type=str, choices=['fit', 'validate', 'test', 'predict'])  # Lightning modes
    parser.add_argument('--architecture', default='wip', type=str, choices=['wip', 'wip_simple', 'unet', 'unetr', 'swin_unetr'])
    parser.add_argument('--architecture_wip', default='deep', type=str, choices=['deep'])  # variants of wip architecture
    parser.add_argument('--check_val_every_n_epoch_', default=10, type=int)
    parser.add_argument('--check_val_every_n_epoch_downstream', default=5, type=int)
    parser.add_argument('--plot_interval_train', default=50, type=int)
    parser.add_argument('--plot_interval_val', default=50, type=int)  # should be 1 or a multiple of check_val_every_n_epoch for regular plots
    parser.add_argument('--plot_interval_test', default=1, type=int)

    # Parameter parsing
    parser = pl.Trainer.add_argparse_args(parser)
    hparams_tmp = parser.parse_known_args()[0]
    cls_dm = BasicDataModule
    if hparams_tmp.architecture.lower() == 'wip':
        cls_model = Model
        cls_architecture = Architecture
    elif hparams_tmp.architecture.lower() == 'wip_simple' or hparams_tmp.architecture.lower() in ['unet', 'unetr', 'swin_unetr']:
        cls_model = ModelSimple
        cls_architecture = ArchitectureSimple
    else:
        raise NotImplementedError(f'{hparams_tmp.architecture} is not a valid architecture.')
    parser = cls_dm.add_data_specific_args(parser)
    parser = cls_model.add_model_specific_args(parser)
    parser = cls_architecture.add_model_specific_args(parser)
    parser = aws.add_aws_specific_args(parser)
    parser = callbacks.add_callback_specific_args(parser)
    hparams = parser.parse_args()

    # fetch and set ckpt if run_name is given (and no ckpt path is passed)
    if hparams.ckpt is None and hparams.ckpt_run_name and not hparams.ckpt_run_name.isspace():
        hparams.ckpt = aws.fetch_ckpt(hparams)

    main(hparams, cls_dm, cls_model)
