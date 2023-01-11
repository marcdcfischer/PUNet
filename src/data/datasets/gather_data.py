from typing import Union, Dict, Tuple
from argparse import ArgumentParser, Namespace
import pathlib as plb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import math


def _generate_splits(df,
                     domains: Tuple[str, ...] = ('tcia', 'btcv'),
                     max_subjects_train: int = -1,
                     num_annotated: int = -1,
                     ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
                     shuffle: bool = True,
                     random_state: int = 1,
                     verbose: bool = True):

    assert sum(ratios) == 1.
    patients = df['names'].unique()
    patients_all = [[x_ for x_ in patients if domain_ in x_] for domain_ in domains]
    dfs_train, dfs_val, dfs_test = list(), list(), list()
    for idx_, patients_ in enumerate(patients_all):
        if len(patients_) > 0:
            patients_train_val, patients_test = train_test_split(patients_, test_size=ratios[2], shuffle=shuffle, random_state=random_state)
            patients_train, patients_val = train_test_split(patients_train_val, test_size=ratios[1] / (1 - ratios[2]), shuffle=shuffle, random_state=random_state)

            if max_subjects_train > 0:  # hard cut of amount of training data
                patients_train = patients_train[:max_subjects_train]

            # Limit training data to percentage of available data. patients_train are not reordered again (since they are already shuffled)
            n_patients_train_full = len(patients_train)
            if num_annotated > 0:
                assert num_annotated <= len(patients_train)  # Otherwise you're trying to grab to many subjects
                patients_train = patients_train[:num_annotated]
            print(f'Using {len(patients_train)} subjects of the available {n_patients_train_full} (~{(len(patients_train) / n_patients_train_full) * 100.} %) for training.')

            dfs_train.append(df[df['names'].isin(patients_train)])
            dfs_val.append(df[df['names'].isin(patients_val)])
            dfs_test.append(df[df['names'].isin(patients_test)])
    df_train = pd.concat(dfs_train)
    df_val = pd.concat(dfs_val)
    df_test = pd.concat(dfs_test)

    if verbose:
        print(f'Split dataset into train, val and test of length: {len(df_train)}, {len(df_val)}, {len(df_test)}.')
        print(f'Training on \n{df_train}.')
        print(f'Validating on \n{df_val}.')
        print(f'Testing on \n{df_test}')

    return df_train, df_val, df_test


def _mask_domains(df,
                  valid_choices: str,
                  modality: str = 't2spir',
                  verbose: bool = True):
    if modality:
        if modality not in valid_choices:
            raise ValueError(f'The modality {modality} is not a valid choice of {valid_choices}')

        # Mask selected modality
        df.loc[df['domains'].str.contains(modality), 'annotated'] = False  # TODO: Suppress warning / replace with more safe change?

        if verbose:
            print(f'Masked modality {modality}.')
            print(f'Masked {df[df["annotated"] == False]["frames"].values} as non-annotated.')
            print(f'Left {df[df["annotated"] == True]["frames"].values} unchanged.')
    else:
        print(f'No masking is performed.')

    return df
