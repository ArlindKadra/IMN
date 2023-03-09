from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import openml
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def prepare_data_for_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    augmentation_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    # Shuffle the data
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > augmentation_prob:
        lam = 1
    else:
        # Generate the mixup mask per example and feature
        for i in range(x.size(0)):
            cut_column_indices = torch.as_tensor(
                np.random.choice(
                    range(x.size(1)),
                    max(1, np.int32(x.size(1) * lam)),
                    replace=False,
                ),
                dtype=torch.int64,
            )

            x[i, cut_column_indices] = x_shuffled[i, cut_column_indices]

    return x, y, y_shuffled, lam


def prepare_data_for_mixup(
    x: torch.Tensor,
    y: torch.Tensor, numerical_features: List,
    augmentation_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    # Shuffle the data
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > augmentation_prob:
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
            cut_column_indices = np.random.choice(
                range(x.size(1)),
                max(1, np.int32(x.size(1) * lam)),
                replace=False,
            )

            cut_cat_indices = [i for i in cut_column_indices if i not in numerical_features]
            cut_numerical_indices = [i for i in cut_column_indices if i in numerical_features]

            if len(cut_cat_indices) > 0:

                cut_cat_indices = torch.as_tensor(
                    cut_cat_indices,
                    dtype=torch.int64,
                )
                x[i, cut_cat_indices] = -1

            if len(cut_numerical_indices) > 0:
                cut_numerical_indices = torch.as_tensor(
                    cut_numerical_indices,
                    dtype=torch.int64,
                )

                x[i, cut_numerical_indices] = 0

    return x, y, y_shuffled, lam

def fgsm_attack(x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, criterion, cut_mix_prob: float, epsilon: float) -> torch.Tensor:

    if np.random.rand() > cut_mix_prob:
        return x
    else:
        # copy tensor to avoid changing the original one
        x = x.clone().detach().requires_grad_(True)

        # perform the attack
        outputs = model(x)
        cost = criterion(outputs, y)

        grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]

        adv_data = x + epsilon * grad.sign()

    return adv_data


def augment_data(x: torch.Tensor, y: torch.Tensor, numerical_features: List, model, criterion, augmentation_prob: float = 0.5) -> Tuple:

    augmentation_types = {
        1: "mixup",
        2: "cutout",
        3: "cutmix",
        4: "fgsm",
    }

    augmentation_type = augmentation_types[np.random.randint(1, 5)]
    if augmentation_type == "cutmix":
        return prepare_data_for_cutmix(x, y, augmentation_prob)
    elif augmentation_type == "mixup":
        return prepare_data_for_mixup(x, y, numerical_features, augmentation_prob)
    elif augmentation_type == "cutout":
        return prepare_data_for_cutout(x, y, numerical_features, augmentation_prob)
    elif augmentation_type == "fgsm":
        return x, fgsm_attack(x, y, model, criterion, augmentation_prob, 0.007), y, y, 0.5
    else:
        raise ValueError("The augmentation type must be one of 'cutmix', 'mixup' or 'cutout'")

def preprocess_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    categorical_indicator: List,
    attribute_names: List,
    test_split_size=0.2,
    seed=11,
) -> Dict:

    imputer = SimpleImputer(missing_values=np.nan, strategy='constant')
    imputer.fit(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_split_size,
        random_state=seed,
        stratify=y,
    )

    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    numerical_features = [i for i in range(len(categorical_indicator)) if not categorical_indicator[i]]
    categorical_features = [i for i in range(len(categorical_indicator)) if categorical_indicator[i]]

    dataset_preprocessors = []
    if len(numerical_features) > 0:
        numerical_preprocessor = ('numerical', StandardScaler(), numerical_features)
        dataset_preprocessors.append(numerical_preprocessor)
    if len(categorical_features) > 0:
        categorical_preprocessor = ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        dataset_preprocessors.append(categorical_preprocessor)

    column_transformer = ColumnTransformer(
        dataset_preprocessors,
        remainder='drop',
    )
    column_transformer.fit(X_train)
    X_train = column_transformer.transform(X_train)
    X_test = column_transformer.transform(X_test)

    # scikit learn label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    if len(numerical_features) > 0:
        new_categorical_indicator = [False] * len(numerical_features)
        new_attribute_names = [attribute_names[i] for i in numerical_features]
    else:
        new_categorical_indicator = []
        new_attribute_names = []

    if len(categorical_features) > 0:
        new_categorical_indicator.extend([True] * len(categorical_features))
        new_attribute_names.extend([attribute_names[i] for i in categorical_features])

    info_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'categorical_indicator': new_categorical_indicator,
        'attribute_names': new_attribute_names,
    }

    return info_dict
def get_dataset(dataset_id: int, test_split_size=0.2, seed=11) -> Dict:

    # Get the data
    dataset = openml.datasets.get_dataset(dataset_id)
    dataset_name = dataset.name
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    info_dict = preprocess_dataset(
        X,
        y,
        categorical_indicator,
        attribute_names,
        test_split_size=test_split_size,
        seed=seed,
    )
    info_dict['dataset_name'] = dataset_name
    return info_dict