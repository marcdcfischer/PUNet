from torch.utils.data import WeightedRandomSampler, RandomSampler  # Would this (random_split) be a good option for ddp?
from argparse import ArgumentParser, Namespace
from src.data.distributed_wrapper import DistributedSamplerWrapper
import torch
import pytorch_lightning as pl
import torchio as tio
from typing import Union, Dict, Optional
import numpy as np

from src.data.transforms_monai import generate_transforms, generate_test_transforms
from src.data.conversion_monai import convert_subjects
# from src.data.datasets.gather_tiny_ixi import gather_data
from src.data.datasets import gather_tcia_btcv, gather_ctorg
from monai.data import DataLoader, Dataset  # Wrapper around torch DataLoader and Dataset


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf  # Model and DataModule hparams are now the same. So only one save_hyperparameters is allowed
        self.transform_train, self.transform_val, self.transform_test, self.transform_test_post = None, None, None, None
        self.df_train, self.df_val, self.df_test = None, None, None,
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self):
        # Nothing to do here - best to do it offline
        pass

    def _get_max_shape_train(self, ds_subjects: Dataset):
        shapes = np.array([crop_['image'].shape for sub_ in ds_subjects for crop_ in sub_])
        return shapes.max(axis=0)

    def _get_max_shape_val(self, ds_subjects: Dataset):
        shapes = np.array([sub_['image'].shape for sub_ in ds_subjects])
        return shapes.max(axis=0)

    def _get_foreground_background_ratio(self, ds_subjects: Dataset):
        ratios = np.zeros((self.conf.out_channels - 1,))
        for sub_ in ds_subjects:
            background_ = (sub_['label'] == 0.).float().count_nonzero().item()
            for idx_ in range(1, self.conf.out_channels):
                ratios[idx_ - 1] += (sub_['label'] == idx_).float().count_nonzero().item() / background_
        ratios /= len(ds_subjects)
        ratios_inv = 1 / ratios
        ratios_bound = np.concatenate([[0], self.conf.additive_alpha_factor * ratios_inv], axis=0)
        return ratios_bound

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:

            # Data gathering / preparation
            if self.conf.dataset.lower() == 'tcia_btcv':
                self.df_train, self.df_val, self.df_test = gather_tcia_btcv.generate_dataframes(self.conf)
            elif self.conf.dataset.lower() == 'ctorg':
                self.df_train, self.df_val, self.df_test = gather_ctorg.generate_dataframes(self.conf)
            else:
                raise NotImplementedError()
            dict_subjects_train, content_keys, aux_keys = convert_subjects(self.df_train)
            dict_subjects_val, _, _ = convert_subjects(self.df_val)
            dict_subjects_test, _, _ = convert_subjects(self.df_test)
            self.transform_train, self.transform_val = generate_transforms(patch_size_students=self.conf.patch_size_students,
                                                                           patch_size_teacher=self.conf.patch_size_teacher,
                                                                           content_keys=content_keys,
                                                                           aux_keys=aux_keys,
                                                                           num_samples=self.conf.num_samples,
                                                                           n_transforms=self.conf.num_transforms,
                                                                           orientation='xy')  # Cropping or padding screws with the coord grid; so prepare data beforehand properly.
            self.transform_test = generate_test_transforms(content_keys=content_keys,
                                                           aux_keys=aux_keys)
            transform_train_zy, _ = generate_transforms(patch_size_students=self.conf.patch_size_students,
                                                        patch_size_teacher=self.conf.patch_size_teacher,
                                                        content_keys=content_keys,
                                                        aux_keys=aux_keys,
                                                        num_samples=self.conf.num_samples,
                                                        n_transforms=self.conf.num_transforms,
                                                        orientation='zy')
            transform_train_xz, _ = generate_transforms(patch_size_students=self.conf.patch_size_students,
                                                        patch_size_teacher=self.conf.patch_size_teacher,
                                                        content_keys=content_keys,
                                                        aux_keys=aux_keys,
                                                        num_samples=self.conf.num_samples,
                                                        n_transforms=self.conf.num_transforms,
                                                        orientation='xz')

            # Datasets
            self.ds_train = Dataset(data=dict_subjects_train, transform=self.transform_train)  # (monai's) CacheDataset may bring some speed up (for deterministic transforms)
            if self.conf.orientation_swap:
                self.ds_train_zy = Dataset(data=dict_subjects_train, transform=transform_train_zy)
                self.ds_train_xz = Dataset(data=dict_subjects_train, transform=transform_train_xz)
            self.ds_val = Dataset(data=dict_subjects_val, transform=self.transform_val)
            self.ds_test = Dataset(data=dict_subjects_test, transform=self.transform_test)

            recalc_ratios = False  # atm hardcoded
            if recalc_ratios:
                self.ds_train_dummy = Dataset(data=dict_subjects_train, transform=self.transform_test)
                additive_alpha = self._get_foreground_background_ratio(self.ds_train_dummy)
                print(f'Recalced additive alpha based on foreground / background ratio: {additive_alpha}')

            max_shape_train = self._get_max_shape_train(self.ds_train)
            max_shape_val = self._get_max_shape_val(self.ds_val)

            print(f'Amount of training samples: {len(self.ds_train)}, and validation samples: {len(self.ds_val)}.')
            print(f'Max shapes for train: {max_shape_train} and val: {max_shape_val}.')
            [print(f'Key: {k_}, Value: {type(v_)}') for k_, v_ in self.ds_train[0][0].items()]
            print(f'Using additive alpha: {self.conf.additive_alpha}.')

        # Assign test dataset for use in dataloader(s)
        elif stage == 'test':
            raise NotImplementedError()  # Atm. only one stage is used for all routines.

        else:
            raise ValueError(f'Stage {stage} is not available.')

    # For self-sup augmentation see: https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/transforms.py#L17-L91
    def train_dataloader(self):
        # See site-packages/pytorch_lightning/trainer for replace_sampler_ddp
        num_samples = int(self.conf.num_samples_epoch / self.conf.num_transforms / self.conf.num_samples)
        print(f'Drawing {num_samples} of {len(self.ds_train)} training samples.')
        if self.conf.weighting:
            shuffle = False  # Exclusive with sampler - nonetheless shuffling can be added
            weights = torch.tensor(self.df_train['weights'].values)
            # num_samples = int(len(self.ds_train) * self.conf.sample_factor)  # Draw half of the dataset (at random), distributed sampler takes care of length for ddp (which splits the data)

            if self.conf.accelerator == 'ddp':
                # Note: neither torch and lightning support distributed versions of most samplers - this is the catalyst implementation  # TODO: Check if a fitting version is now available
                # Lightning takes care of set_epoch for proper shuffling
                sampler = DistributedSamplerWrapper(WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=self.conf.replacement), shuffle=True)
            else:
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=self.conf.replacement)  # should always shuffle (due to random draw)
        else:
            sampler = RandomSampler(self.ds_train, replacement=True, num_samples=num_samples)
        assert (self.conf.batch_size / self.conf.num_transforms / self.conf.num_samples).is_integer()

        loader_xy = DataLoader(self.ds_train,
                               batch_size=self.conf.batch_size // self.conf.num_transforms // self.conf.num_samples,
                               num_workers=self.conf.num_workers,
                               pin_memory=self.conf.pin_memory,
                               shuffle=False,  # Has to be False for given sampler
                               sampler=sampler,
                               # collate_fn=collate_list,
                               drop_last=True)
        loaders = [loader_xy]
        if self.conf.orientation_swap:
            loader_zy = DataLoader(self.ds_train_zy,
                                   batch_size=self.conf.batch_size // self.conf.num_transforms // self.conf.num_samples,
                                   num_workers=self.conf.num_workers,
                                   pin_memory=self.conf.pin_memory,
                                   shuffle=False,  # Has to be False for given sampler
                                   sampler=sampler,
                                   # collate_fn=collate_list,
                                   drop_last=True)
            loader_xz = DataLoader(self.ds_train_xz,
                                   batch_size=self.conf.batch_size // self.conf.num_transforms // self.conf.num_samples,
                                   num_workers=self.conf.num_workers,
                                   pin_memory=self.conf.pin_memory,
                                   shuffle=False,  # Has to be False for given sampler
                                   sampler=sampler,
                                   # collate_fn=collate_list,
                                   drop_last=True)
            loaders.extend([loader_zy, loader_xz])
        return loaders

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=1,  # Note: Can be smaller than batch_size if len(ds_val) < batch_size.
                          num_workers=self.conf.num_workers,
                          pin_memory=self.conf.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=1,
                          num_workers=self.conf.num_workers,
                          pin_memory=self.conf.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=1,
                          num_workers=self.conf.num_workers,
                          pin_memory=self.conf.pin_memory)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--pin_memory', action='store_true')
        parser.add_argument('--replacement', default=True, type=bool)  # Use sampling with replacement if annotation is only sparsely available
        parser.add_argument('--weighting', default=True, type=bool)  # Custom weighting
        parser.add_argument('--sample_factor', default=1., type=float)  # Amount of samples drawn per epoch for weighted sampling
        parser.add_argument('--num_transforms', default=1, type=int)  # Amount of transforms applied to a selected (cropped) image. For > 1, this produces a "positive" pair. Keep it high-ish so crops find some overlapping regions
        parser.add_argument('--num_samples', default=2, type=int)  # Samples per subject
        parser.add_argument('--num_samples_epoch', default=5000, type=int)  # Amount of samples in an epoch
        parser.add_argument('--queue_max_length', default=36, type=int)
        parser.add_argument('--orientation_swap', default=False, type=bool)  # Randomly permute axes
        parser.add_argument('--max_subjects_train', default=-1, type=int)
        parser.add_argument('--dataset', default='tcia_btcv', type=str, choices=['tcia_btcv', 'ctorg'])

        hparams_tmp = parser.parse_known_args()[0]
        if hparams_tmp.dataset.lower() == 'tcia_btcv':
            parser = gather_tcia_btcv.add_data_specific_args(parser)
        elif hparams_tmp.dataset.lower() == 'ctorg':
            parser = gather_ctorg.add_data_specific_args(parser)
        else:
            raise NotImplementedError(f'The selected architecture {hparams_tmp.architecture} is not available.')

        return parser
