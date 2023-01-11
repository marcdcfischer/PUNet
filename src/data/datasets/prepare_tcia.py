import pathlib as plb
import re
import csv
import nibabel as nib
import numpy as np
from sklearn import preprocessing
from src.data.datasets.pre_processing import _rescale
from typing import Tuple


def _process(path_img,
             path_lbl,
             coords,
             dir_out,
             classes: Tuple[int, ...] = (0, 1, 3, 4, 5, 6, 7, 11, 14),
             view: bool = False):
    nii_image, nii_lbl = nib.load(path_img), nib.load(path_lbl)
    np_image = nii_image.get_fdata(dtype=np.float32)
    np_label = np.flip(nii_lbl.get_fdata(dtype=np.float32).round().astype(np.int32), axis=2)
    assert np_image.shape == np_label.shape

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([np_image / 255. * 10., np_label], axis=-1))
        viewer.show()

    # Crop (according to given CSV) +- x
    additional_space = 24
    additional_space_z = 3
    np_image = np_image[max(coords[0] - additional_space, 0):min(coords[1] + additional_space, np_image.shape[0]),
                        max(np_image.shape[1] - coords[3] - additional_space, 0):min(np_image.shape[1] - coords[2] + additional_space, np_image.shape[1]),
                        max(coords[4] - additional_space_z, 0):min(coords[5] + additional_space_z, np_image.shape[2])]
    np_label = np_label[max(coords[0] - additional_space, 0):min(coords[1] + additional_space, np_label.shape[0]),
                        max(np_label.shape[1] - coords[3] - additional_space, 0):min(np_label.shape[1] - coords[2] + additional_space, np_label.shape[1]),
                        max(coords[4] - additional_space_z, 0):min(coords[5] + additional_space_z, np_label.shape[2])]

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([np_image / 255. * 10., np_label], axis=-1))
        viewer.show()

    # Label encoding
    print(f'Selecting labels {classes} out of {np.unique(np_label)}.')
    le = preprocessing.LabelEncoder()
    # Classes: 1. Spleen, 3. L. Kidney, 4. Gallbladder, 5. Esophagus, 6. Liver, 7. Stomach, 11. Pancreas, 14. Duodenum
    # Groups: 1. Spleen, L. Kidney, Gallbladder, Liver, 2. Esophagus, Stomach, Pancreas, Duodenum
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
    nib.save(nii_image_converted, plb.Path(dir_out) / (path_img.name + '.gz'))
    nib.save(nii_lbl_converted, plb.Path(dir_out) / path_lbl.name)


if __name__ == '__main__':
    # data from https://zenodo.org/record/1169361#.YqhgFXhBxhH
    dir_data = '/mnt/SSD_SATA_03/data_med/prompting/multi-organ/'
    dir_out = '/mnt/SSD_SATA_03/data_med/prompting/processed_tcia_fixed_spacing/'
    plb.Path(dir_out).mkdir(parents=True, exist_ok=True)
    target_resolution = (1.25, 1.25, 2.5)
    target_size = (280, 280, -1)
    classes = (0, 1, 3, 4, 5, 6, 7, 11, 14)
    view = False
    overwrite = True

    # Preprocess data - cropping via csv
    crop_coords = dict()
    with open(plb.Path(dir_data) / 'cropping.csv') as file_:
        reader = csv.DictReader(file_)
        for row in reader:
            if row['publisher'] == 'TCIA':
                crop_coords[row['original_id'].zfill(4)] = [int(x_) for x_ in [row['extent_left'], row['extent_right'],
                                                                               row['extent_ant'], row['extent_post'],
                                                                               row['extent_inf'], row['extent_sup']]]  # coords reordered due to wrong entries

    paths_imgs = sorted(list((plb.Path(dir_data) / 'image_tcia_multiorgan').glob('*.nii')))
    paths_lbls = sorted(list((plb.Path(dir_data) / 'label_tcia_multiorgan').glob('*.nii.gz')))
    assert len(paths_imgs) == len(paths_lbls)
    img_numbers = [re.search('\d{4}', str(x_.name)).group(0) for x_ in paths_imgs]
    lbl_numbers = [re.search('\d{4}', str(x_.name)).group(0) for x_ in paths_lbls]
    assert set(img_numbers) == set(lbl_numbers)

    for path_img_, path_lbl_ in zip(paths_imgs, paths_lbls):
        print(f'--- Processing {path_img_.name} ---')
        id_ = re.search('\d{4}', str(path_img_.name)).group(0)
        coords_ = crop_coords[id_]

        if overwrite:
            _process(path_img=path_img_,
                     path_lbl=path_lbl_,
                     coords=coords_,
                     dir_out=dir_out,
                     classes=classes,
                     view=view)

            _rescale(path_images=[str(plb.Path(dir_out) / (path_img_.name + '.gz'))],
                     path_mask=str(plb.Path(dir_out) / path_lbl_.name),
                     target_resolution=target_resolution,
                     target_size=target_size,
                     view=view)
