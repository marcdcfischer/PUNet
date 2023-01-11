from typing import Optional, Tuple, List, Type
import monai.transforms as mtransforms
import itertools
import pathlib as plb


# See https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb for a recent example
def generate_transforms(patch_size_students: List[Tuple[int, int, int]],
                        patch_size_teacher: Optional[Tuple[int, int, int]] = None,
                        content_keys: Optional[List[str]] = None,
                        aux_keys: Optional[List[str]] = None,
                        num_samples: int = 2,  # Different slice crops per volume
                        n_transforms: int = 1,  # Different transforms of cropped volumes
                        orientation: str = 'xy',
                        intensity_inversion: bool = False,
                        augmentation_mode: str = '2d',
                        shape_z: int = 1,
                        orientation_augmentation: bool = False):
    patch_size_teacher = patch_size_teacher if patch_size_teacher is not None else patch_size_students[0]
    if content_keys is None:
        content_keys = ['image', 'label']

    # Original keys are considered student keys
    content_keys_trafos = [[key_ + f'_trafo{str(idx_trafo)}' for key_ in content_keys] for idx_trafo in range(1, n_transforms)]
    aux_keys_trafos = [[key_ + f'_trafo{str(idx_trafo)}' for key_ in aux_keys] for idx_trafo in range(1, n_transforms)]
    content_keys_all_student_first = [content_keys] + content_keys_trafos  # list of lists (so every sample can be transformed independently)
    aux_keys_all_student_first = [aux_keys] + aux_keys_trafos

    # Students extended keys (e.g. smaller variant)
    # Following scheme [idx_student][idx_trafo][key]
    n_students = len(patch_size_students)
    content_keys_students = [content_keys]
    content_keys_all_students = [content_keys_all_student_first]
    aux_keys_all_students = [aux_keys_all_student_first]
    for idx_student in range(0, n_students - 1):
        content_keys_students.append([key_ + f'_var{str(idx_student)}' for key_ in content_keys])  # Student content keys without trafo
        content_keys_all_students.append([[key_ + f'_var{str(idx_student)}' for key_ in content_keys_] for content_keys_ in content_keys_all_student_first])  # Student content keys with trafo
        aux_keys_all_students.append([[key_ + f'_var{str(idx_student)}' for key_ in aux_keys_] for aux_keys_ in aux_keys_all_student_first])

    # Teacher keys
    content_keys_teacher = [key_ + '_teacher' for key_ in content_keys]  # [key]
    content_keys_all_teacher = [[key_ + '_teacher' for key_ in content_keys_] for content_keys_ in content_keys_all_student_first]  # [idx_trafo][key]
    aux_keys_all_teacher = [[key_ + '_teacher' for key_ in aux_keys_] for aux_keys_ in aux_keys_all_student_first]  # [idx_trafo][key]

    # All keys
    content_keys_all = content_keys_all_students + [content_keys_all_teacher]  # [idx_student / teacher][idx_trafo][key]
    aux_keys_all = aux_keys_all_students + [aux_keys_all_teacher]  # [idx_student / teacher][idx_trafo][key]

    # Pre-processing
    transform_train = mtransforms.Compose([
        # Image loading, normalization and (sub-)selection
        mtransforms.LoadImaged(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),  # Exclude tensors from loading
        mtransforms.EnsureChannelFirstd(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),   # Grid already has a channel (and does not have metadata for a check)
        # mtransforms.NormalizeIntensityd(keys=[x_ for x_ in content_keys if 'image' in str(x_)])
        mtransforms.ScaleIntensityRanged(keys=[x_ for x_ in content_keys if 'image' in str(x_)], a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),  # CT only
    ])

    # Rotate orientation
    if orientation == 'xy':
        pass
    elif orientation == 'zy':
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.Rotate90d(keys=content_keys, k=1, spatial_axes=(0, 2)),  # Rotate xz (as orientation augmentation) so resulting slices contain zy
        ])
    elif orientation == 'xz':
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.Rotate90d(keys=content_keys, k=1, spatial_axes=(1, 2)),  # Rotate yz (as orientation augmentation) so resulting slices contain xz
        ])
    else:
        raise ValueError(f'Orientation {orientation} is not a valid choice.')

    # Generate samples
    transform_train = mtransforms.Compose([
        transform_train,
        mtransforms.RandSpatialCropSamplesd(keys=content_keys, roi_size=(patch_size_teacher[0], patch_size_teacher[1], patch_size_teacher[2]), random_center=True, random_size=False, num_samples=num_samples),  # Generates num_samples different slices
        # mtransforms.Resized(keys=content_keys, spatial_size=(256, 256, 1), mode=['nearest' if 'label' in key_ else 'trilinear' for key_ in content_keys]),  # Should usually be done in (offline) pre-processing
        mtransforms.CopyItemsd(keys=content_keys + aux_keys, times=n_transforms - 1, names=list(itertools.chain(*(content_keys_trafos + aux_keys_trafos)))) if n_transforms > 1 else mtransforms.Compose([]),  # Copies selected slices (and auxiliary info) for different augmentations
        mtransforms.CopyItemsd(keys=list(itertools.chain(*(content_keys_all_student_first + aux_keys_all_student_first))), times=1,
                               names=list(itertools.chain(*(content_keys_all_teacher + aux_keys_all_teacher)))),
    ])

    # Add further student samples
    for idx_student in range(1, n_students):
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.CopyItemsd(keys=list(itertools.chain(*(content_keys_all_student_first + aux_keys_all_student_first))), times=1,
                                   names=list(itertools.chain(*(content_keys_all_students[idx_student] + aux_keys_all_students[idx_student])))),
        ])

    # Masking - see https://github.com/Project-MONAI/tutorials/tree/master/self_supervised_pretraining for more
    # Only applied to student.
    # dropout_holes = True -> replaces values inside region. dropout_holes = False -> replaces values outside region
    # Students (large and small) specific augmentations
    for idx_student in range(n_students):
        for idx_trafo in range(n_transforms):
            transform_train = mtransforms.Compose([
                transform_train,
                mtransforms.OneOf([
                    mtransforms.Compose([]),  # Dummy for applying nothing
                    mtransforms.RandCoarseDropoutd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=1.0,
                                                   dropout_holes=True, holes=1, max_holes=3, spatial_size=5, max_spatial_size=20),  # spatial_size=20, max_spatial_size=40 used for a visualization example
                    mtransforms.RandCoarseDropoutd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=1.0,
                                                   dropout_holes=False, holes=6, spatial_size=48, max_spatial_size=96),  # Holes are aggregated prior to outer fill.
                    mtransforms.RandCoarseShuffled(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=1.0,
                                                   holes=1, max_holes=3, spatial_size=5, max_spatial_size=20),
                ], weights=(0.7, 0.1, 0.1, 0.1))  # (0.0, 1.0, 0.0, 0.0) used for a visualization example
            ])

    # Augmentations on (pre-)crop
    # Note: Different samples (from CropSamples) are handled automatically, as they are internally stored in a list that is later collated.
    #       Only the ones generated by CopyItem need to be taken care off.
    # Students (large and small) are augmented differently
    for idx_student in range(n_students):
        for idx_trafo in range(n_transforms):
            transform_train = mtransforms.Compose([
                transform_train,
                mtransforms.RandBiasFieldd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=0.1),
                mtransforms.RandStdShiftIntensityd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=0.1, factors=(0.0, 0.25)),
                mtransforms.RandAdjustContrastd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=0.1),
                mtransforms.RandScaleIntensityd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=0.1, factors=-2.) if intensity_inversion else mtransforms.Compose([]),  # Invert image (v = v * (1 + factor))
                mtransforms.RandHistogramShiftd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=0.1, num_control_points=(8, 12)),  # Shifts around image histogram
                # mtransforms.Rand3DElastic(keys=content_keys_transformed[idx_]),
                mtransforms.RandAffined(keys=content_keys_all_students[idx_student][idx_trafo],
                                        prob=0.8,
                                        rotate_range=(0., 0., 0.4),  # 2D rotate in radians
                                        shear_range=(0.025, 0., 0.025, 0., 0., 0.),  # 2D shear
                                        # translate_range=(0.25, 0.25, 0.0), already handled by random crop
                                        scale_range=((-0.25, 0.25), (-0.25, 0.25), (0., 0.)),  # 2D scaling. For some reason they add 1.0 internally ...
                                        mode=['nearest' if 'label' in key_ or 'pseudo' in key_ else 'bilinear' for key_ in content_keys_all_students[idx_student][idx_trafo]],
                                        padding_mode='reflection'),
            ])
    # Teacher
    for idx_trafo in range(n_transforms):
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.RandBiasFieldd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=0.1),
            mtransforms.RandStdShiftIntensityd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=0.1, factors=(0.0, 0.1)),
            mtransforms.RandAdjustContrastd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=0.1),
            mtransforms.RandHistogramShiftd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=0.1, num_control_points=12),  # Fixed number of control points
            mtransforms.RandAffined(keys=content_keys_all_teacher[idx_trafo],
                                    prob=0.5,
                                    rotate_range=(0., 0., 0.4),
                                    shear_range=(0.025, 0., 0.025, 0., 0., 0.),
                                    scale_range=((-0.1, 0.1), (-0.1, 0.1), (0., 0.)),
                                    mode=['nearest' if 'label' in key_ or 'pseudo' in key_ else 'bilinear' for key_ in content_keys_all_teacher[idx_trafo]],
                                    padding_mode='reflection'),
        ])

    # Final conversion
    # Students
    for idx_student in range(n_students):
        for idx_trafo in range(n_transforms):
            transform_train = mtransforms.Compose([
                transform_train,
                mtransforms.RandSpatialCropd(keys=content_keys_all_students[idx_student][idx_trafo], roi_size=patch_size_students[idx_student], random_center=True, random_size=False),  # Crop center of augmented patch
                mtransforms.SpatialPadd(keys=content_keys_all_students[idx_student][idx_trafo], spatial_size=patch_size_students[idx_student], mode='reflect'),  # Needs to have at least this size. Reflect may lead to undesired (repetitive) patterns. Better options consistent with coords?
                mtransforms.ToTensord(keys=content_keys_all_students[idx_student][idx_trafo])
            ])
    # Teacher
    for idx_trafo in range(n_transforms):
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.RandSpatialCropd(keys=content_keys_all_teacher[idx_trafo], roi_size=patch_size_teacher, random_center=True, random_size=False),  # Crop center of augmented patch
            mtransforms.SpatialPadd(keys=content_keys_all_teacher[idx_trafo], spatial_size=patch_size_teacher, mode='reflect'),  # Needs to have at least this size. Reflect may lead to undesired (repetitive) patterns. Better options consistent with coords?
            mtransforms.ToTensord(keys=content_keys_all_teacher[idx_trafo])
        ])

    # Join transformed elements (along channel dim) - so it doesn't need to perform within the training / validation routine
    # Students
    for idx_student in range(n_students):
        for key_, keys_student_ in zip(content_keys, content_keys_students[idx_student]):
            applied_student_keys_ = [x_ for x_ in list(itertools.chain(*content_keys_all_students[idx_student])) if key_ in str(x_)]
            if len(applied_student_keys_) > 0:
                transform_train = mtransforms.Compose([transform_train, mtransforms.ConcatItemsd(keys=applied_student_keys_, name=keys_student_)])
    # Teacher
    for key_, key_teacher_ in zip(content_keys, content_keys_teacher):
        applied_teacher_keys = [x_ for x_ in list(itertools.chain(*content_keys_all_teacher)) if key_ in str(x_)]
        if len(applied_teacher_keys) > 0:
            transform_train = mtransforms.Compose([transform_train, mtransforms.ConcatItemsd(keys=applied_teacher_keys, name=key_teacher_)])

    # Discard obsolete additional keys and meta data (of additional trafos and variants)
    transform_train = mtransforms.Compose([
        transform_train,
        mtransforms.SelectItemsd(keys=list(itertools.chain(*content_keys_students)) + content_keys_teacher + aux_keys)
    ])

    # Validation
    transform_val = mtransforms.Compose([
        mtransforms.LoadImaged(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        mtransforms.EnsureChannelFirstd(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        # mtransforms.NormalizeIntensityd(keys=[x_ for x_ in content_keys if 'image' in str(x_)]),
        mtransforms.ScaleIntensityRanged(keys=[x_ for x_ in content_keys if 'image' in str(x_)], a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),  # CT only
        # mtransforms.RandSpatialCropSamplesd(keys=content_keys, roi_size=patch_size_teacher, random_center=True, random_size=False, num_samples=10),
        # mtransforms.SpatialPadd(keys=content_keys, spatial_size=patch_size_teacher, mode='constant'),  # Needs to have at least this size. Reflect may lead to undesired (repetitive) patterns. Better options consistent with coords?
        mtransforms.ToTensord(keys=content_keys),
        mtransforms.SelectItemsd(keys=content_keys + aux_keys)  # Discard (currently) unused meta data
    ])

    return transform_train, transform_val


def generate_test_transforms(content_keys: Optional[List[str]] = None,
                             aux_keys: Optional[List[str]] = None):
    if content_keys is None:
        content_keys = ['image', 'label']

    # Validation
    transform_test = mtransforms.Compose([
        mtransforms.LoadImaged(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        mtransforms.EnsureChannelFirstd(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        # mtransforms.NormalizeIntensityd(keys=[x_ for x_ in content_keys if 'image' in str(x_)]),
        mtransforms.ScaleIntensityRanged(keys=[x_ for x_ in content_keys if 'image' in str(x_)], a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),  # CT only
        mtransforms.ToTensord(keys=content_keys),
        # mtransforms.SelectItemsd(keys=content_keys + aux_keys)  # Discard (currently) unused meta data
    ])

    return transform_test


def generate_test_post_transforms(output_dir: str,
                                  output_postfix: str,
                                  transform_test: mtransforms.InvertibleTransform,
                                  n_classes: Optional[int] = None):

    # Create output directory (if it doesn't exist)
    plb.Path(output_dir).mkdir(parents=True, exist_ok=True)

    transform_test_post = mtransforms.Compose([
        mtransforms.EnsureTyped(keys="pred"),
        mtransforms.Invertd(
            keys="pred",
            transform=transform_test,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        mtransforms.AsDiscreted(keys="pred", argmax=True, to_onehot=n_classes),
        mtransforms.SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=output_dir, output_postfix=output_postfix, resample=False),
    ])

    return transform_test_post
