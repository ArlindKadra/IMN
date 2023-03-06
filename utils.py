from typing import List, Tuple

import numpy as np
import torch


def prepare_data_for_cutmix(x: torch.Tensor, y: torch.Tensor, numerical_features: List, cut_mix_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    # Shuffle the data
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > cut_mix_prob:
        lam = 1
    else:
        # Generate the mixup mask per example and feature
        for i in range(x.size(0)):
            cut_column_indices = torch.as_tensor(
                np.random.choice(
                    numerical_features,
                    max(1, np.int32(len(numerical_features) * lam)),
                    replace=False,
                ),
                dtype=torch.int64,
            )

            x[i, cut_column_indices] = x_shuffled[i, cut_column_indices]

    return x, y, y_shuffled, lam


def prepare_data_for_mixup(x: torch.Tensor, y: torch.Tensor, numerical_features: List, cut_mix_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    # Shuffle the data
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > cut_mix_prob:
        lam = 1
    else:
        # Generate the mixup mask per example and feature
        for i in range(x.size(0)):
            cut_column_indices = torch.as_tensor(
                np.random.choice(
                    numerical_features,
                    max(1, np.int32(len(numerical_features) * lam)),
                    replace=False,
                ),
                dtype=torch.int64,
            )

            x[i, cut_column_indices] = lam * x[i, cut_column_indices] + (1. - lam) * x_shuffled[i, cut_column_indices]

    return x, y, y_shuffled, lam


def prepare_data_for_cutout(x: torch.Tensor, y: torch.Tensor, numerical_features: List, cut_mix_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    # Shuffle the data
    indices = torch.randperm(x.size(0))
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > cut_mix_prob:
        lam = 1
    else:
        # Generate the mixup mask per example and feature
        for i in range(x.size(0)):
            cut_column_indices = torch.as_tensor(
                np.random.choice(
                    numerical_features,
                    max(1, np.int32(len(numerical_features) * lam)),
                    replace=False,
                ),
                dtype=torch.int64,
            )

            x[i, cut_column_indices] = 0

    return x, y, y_shuffled, lam

def augment_data(x: torch.Tensor, y: torch.Tensor, numerical_features: List, cut_mix_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    augmentation_types = {
        1: "mixup",
        2: "cutmix",
        3: "cutout",
    }

    cut_mix_type = augmentation_types[np.random.randint(1, 4)]

    if cut_mix_type == "cutmix":
        return prepare_data_for_cutmix(x, y, numerical_features, cut_mix_prob)
    elif cut_mix_type == "mixup":
        return prepare_data_for_mixup(x, y, numerical_features, cut_mix_prob)
    elif cut_mix_type == "cutout":
        return prepare_data_for_cutout(x, y, numerical_features, cut_mix_prob)
    else:
        raise ValueError("The augmentation type must be one of 'cutmix', 'mixup' or 'cutout'")
