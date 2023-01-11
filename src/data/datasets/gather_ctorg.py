from typing import Union, Dict, Tuple
from argparse import ArgumentParser, Namespace
import pathlib as plb
import pandas as pd
import numpy as np
from typing import List
from src.data.datasets.gather_data import _generate_splits, _mask_domains


def generate_dataframes(conf: Union[Dict, Namespace]):

    df_dirs = _gather_data(dir_images=conf.dir_images,
                           dir_masks=conf.dir_masks,
                           domains=conf.domains)

    df_train, df_val, df_test = _generate_splits(df_dirs,
                                                 num_annotated=conf.num_annotated,
                                                 domains=conf.domains,
                                                 max_subjects_train=conf.max_subjects_train)

    df_train = _mask_domains(df_train,
                             modality=conf.masked_modality,
                             valid_choices=conf.domains)

    return df_train, df_val, df_test


def _gather_data(dir_images: str,
                 dir_masks: str,
                 domains: List[str]):

    data_dirs = dict()
    for key_ in ['names', 'frames', 'domains', 'images', 'masks']:
        data_dirs[key_] = list()

    if 'ctorg' in domains:
        paths_images_tcia = sorted((plb.Path(dir_images) / 'processed_ctorg').rglob('volume*.nii.gz'))
        paths_masks_tcia = sorted((plb.Path(dir_masks) / 'processed_ctorg').rglob('labels*.nii.gz'))
        data_dirs['names'].extend([x_.name.split('.')[0] + '_ctorg' for x_ in paths_images_tcia])
        data_dirs['frames'].extend([x_.name.split('.')[0] + '_ctorg' for x_ in paths_images_tcia])
        data_dirs['domains'].extend(['ctorg' for _ in paths_images_tcia])
        data_dirs['images'].extend(paths_images_tcia)
        data_dirs['masks'].extend(paths_masks_tcia)

    debug = False
    if debug:
        import matplotlib
        import nibabel as nib
        matplotlib.use('tkagg')
        viewer = nib.viewers.OrthoSlicer3D(np.array(np.stack([nib.load(paths_images_tcia[0]).get_fdata() / 255.,
                                                              nib.load(paths_masks_tcia[0]).get_fdata()], axis=-1)))
        viewer.show()

    df_dirs = pd.DataFrame(data_dirs)
    df_dirs = df_dirs.assign(annotated=True)
    df_dirs = df_dirs.assign(weights=1.0)  # Default weight is 1.0

    return df_dirs


def add_data_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--dir_images', default='/mnt/SSD_SATA_03/data_med/prompting/', type=str)  # EDIT ME
    parser.add_argument('--dir_masks', default='/mnt/SSD_SATA_03/data_med/prompting/', type=str)  # EDIT ME
    # parser.add_argument('--dir_scribbles', default='/mnt/SSD_SATA_03/data_med/scribbles/acdc_scribbles_2020_fixed', type=str)
    # parser.add_argument('--image_size', default=[64, 64, 48], nargs=3, type=int)
    parser.add_argument('--n_students', default=2, type=int)
    parser.add_argument('--patch_size_students', default="224,224,1; 160,160,1", type=list_of_tupels)
    parser.add_argument('--patch_size_teacher', default=[256, 256, 1], nargs=3, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=6, type=int)
    parser.add_argument('--masked_modality', default='', type=str, choices=['ctorg'])
    parser.add_argument('--domains', default=['ctorg'])  # Present domains. Used e.g. for domain-wise prototypes
    parser.add_argument('--num_annotated', default=-1, type=int)  # Determines amount of annotated subjects available during training. 10 ~ 10%, 19 ~ 20%, 48 ~ 50%.
    parser.add_argument('--additive_alpha', default=[0., 0.06171474, 0.74141834, 0.16770616, 0.00349171, 0.20303888], type=float)  # Additive alpha value based on foreground / background ratio. 0 for background.
    parser.add_argument('--additive_alpha_factor', default=0.01, type=float)  # Factor to compress the range of minimal (0.) and maximal additive alpha value
    return parser


def list_of_tupels(args):
    lists_ = [x_.split(',') for x_ in args.replace(' ','').split(';')]
    tuples_ = list()
    for list_ in lists_:
        tuples_.append(tuple([int(x_) for x_ in list_]))
    return tuples_
