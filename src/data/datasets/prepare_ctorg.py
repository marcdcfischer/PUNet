import pathlib as plb
import re
import csv
import nibabel as nib
import numpy as np
from sklearn import preprocessing
from src.data.datasets.pre_processing import _rescale
from typing import Tuple
from p_tqdm import p_map


def _process(path_img,
             path_lbl,
             dir_out,
             classes: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
             view: bool = False):
    nii_image, nii_lbl = nib.load(path_img), nib.load(path_lbl)
    np_image = nii_image.get_fdata(dtype=np.float32)
    np_label = nii_lbl.get_fdata(dtype=np.float32).round().astype(np.int32)
    assert np_image.shape == np_label.shape

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([np_image / 255. * 10., np_label], axis=-1))
        viewer.show()

    if 1 not in np.unique(np_label):
        print('Case of missing liver label. Subject will be discarded.')
        return False

    # Label encoding
    print(f'Selecting labels {classes} out of {np.unique(np_label)}.')
    le = preprocessing.LabelEncoder()
    # Classes: 0: Background (None of the following organs), 1: Liver, 2: Bladder, 3: Lungs, 4: Kidneys, 5: Bone, 6: Brain
    # Groups: 1. Liver, Bladder, Kidneys, 2. Lungs, Bone
    le.classes_ = classes
    mask = np.full_like(np_label, fill_value=1).astype(bool)
    for class_ in classes:
        mask = np.logical_and(mask, np_label != class_)
    np_label[mask] = 0
    np_label = le.transform(np_label.reshape(-1)).reshape(np_label.shape)

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([np_image / 255. * 10., np_label], axis=-1))
        viewer.show()

    nii_image_converted = nib.Nifti1Image(np_image, nii_image.affine, nii_image.header)
    nii_lbl_converted = nib.Nifti1Image(np_label,  nii_lbl.affine, nii_lbl.header)
    nib.save(nii_image_converted, plb.Path(dir_out) / path_img.name)
    nib.save(nii_lbl_converted, plb.Path(dir_out) / path_lbl.name)

    return True


if __name__ == '__main__':
    # data from https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations#61080890bcab02c187174a288dbcbf95d26179e8
    dir_data = '/mnt/SSD_SATA_03/data_med/prompting/OrganSegmentations/'
    dir_out = '/mnt/SSD_SATA_03/data_med/prompting/processed_ctorg_fixed_spacing/'
    plb.Path(dir_out).mkdir(parents=True, exist_ok=True)
    target_resolution = (1.25, 1.25, 2.5)
    target_size = (280, 280, -1)
    classes = (0, 1, 2, 3, 4, 5)
    view = False
    overwrite = True

    paths_imgs = sorted(list(plb.Path(dir_data).glob('volume*.nii.gz')))
    paths_lbls = sorted(list(plb.Path(dir_data).glob('labels*.nii.gz')))
    assert len(paths_imgs) == len(paths_lbls)
    img_numbers = [re.search('\d+', str(x_.name)).group(0) for x_ in paths_imgs]
    lbl_numbers = [re.search('\d+', str(x_.name)).group(0) for x_ in paths_lbls]
    assert set(img_numbers) == set(lbl_numbers)

    def _process_single(path_img_, path_lbl_):
        print(f'--- Processing {path_img_.name} ---')
        if overwrite:
            valid = _process(path_img=path_img_,
                             path_lbl=path_lbl_,
                             dir_out=dir_out,
                             classes=classes,
                             view=view)

            if valid:
                _rescale(path_images=[str(plb.Path(dir_out) / path_img_.name)],
                         path_mask=str(plb.Path(dir_out) / path_lbl_.name),
                         target_resolution=target_resolution,
                         target_size=target_size,
                         view=view)

    p_map(_process_single,
          paths_imgs,
          paths_lbls,
          num_cpus=0.5)
