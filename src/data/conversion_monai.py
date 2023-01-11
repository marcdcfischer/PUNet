import torch
import torchio as tio
import pathlib as plb
from typing import Union, Optional, Tuple
import pandas as pd
import nibabel as nib
import numpy as np


def convert_subjects(df_data: pd.DataFrame,
                     with_coords: bool = True,
                     visualize: bool = False):

    subject_dicts = list()
    content_keys = ['image', 'coord_grid']
    aux_keys = ['name', 'domain', 'frame', 'annotated', 'age']
    for idx_, row_ in df_data.iterrows():
        subject_dicts.append(
            {
                'image': row_['images'],
                'coord_grid': _gen_mesh_grid(nib.load(row_['images']).shape) if with_coords else 0,
                'name': row_['names'],
                'frame': row_['frames'],  # NOT 'names' as they might not be unique!
                'domain': row_['domains'],
                'annotated': row_['annotated'],
                'age': 0,  # Or whatever you might want to add
            }
        )
        # Add available annotations
        if 'masks' in row_.keys() and row_['masks'] is not None:
            subject_dicts[-1]['label'] = row_['masks']
            if 'label' not in content_keys:
                content_keys.append('label')
        if 'pseudos' in row_.keys() and row_['pseudos'] is not None:
            subject_dicts[-1]['pseudo'] = row_['pseudos']
            if 'pseudo' not in content_keys:
                content_keys.append('pseudo')
        if 'scribbles' in row_.keys() and row_['scribbles'] is not None:
            subject_dicts[-1]['scribbles'] = row_['scribbles']
            if 'scribbles' not in content_keys:
                content_keys.append('scribbles')

    return subject_dicts, content_keys, aux_keys


def _gen_mesh_grid(image_size: Tuple[int, int, int]):
    coord_grid = torch.stack(torch.meshgrid(torch.arange(image_size[0]), torch.arange(image_size[1]), torch.arange(image_size[2])), dim=0).float()  # [3, H, W, D]
    coord_grid -= torch.tensor(((image_size[0] - 1) / 2., (image_size[1] - 1) / 2., (image_size[2] - 1) / 2.)).reshape(3, 1, 1, 1)  # Shift coords so center of volume is at [0, 0, 0]. That way misalignments are most severe at the boundaries.
    return np.array(coord_grid)
