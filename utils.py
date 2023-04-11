from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import openml
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        cost = criterion(outputs, y)

        grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]

        adv_data = x + epsilon * grad.sign()

    return adv_data

def random_noise(
    x: torch.Tensor,
    y: torch.Tensor,
    augmentation_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:


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
            x[i, cut_column_indices] = torch.add(x[i, cut_column_indices], (0.1 ** 0.5) * torch.randn(x[i, cut_column_indices].shape).to(x.device))

    return x, y, y, 1


def augment_data(x: torch.Tensor, y: torch.Tensor, numerical_features: List, model, criterion, augmentation_prob: float = 0.5) -> Tuple:

    augmentation_types = {
        1: "mixup",
        2: "cutout",
        3: "cutmix",
        4: "fgsm",
        5: "random_noise",
    }

    if len(numerical_features) == 0:
        augmentation_types = {
            1: "cutout",
            2: "cutmix",
            3: "fgsm",
            4: "random_noise",
        }

    augmentation_type = augmentation_types[np.random.randint(1, len(augmentation_types) + 1)]
    if augmentation_type == "cutmix":
        return prepare_data_for_cutmix(x, y, augmentation_prob)
    elif augmentation_type == "mixup":
        return prepare_data_for_mixup(x, y, numerical_features, augmentation_prob)
    elif augmentation_type == "cutout":
        return prepare_data_for_cutout(x, y, numerical_features, augmentation_prob)
    elif augmentation_type == "fgsm":
        return x, fgsm_attack(x, y, model, criterion, augmentation_prob, 0.007), y, y, 0.5
    elif augmentation_type == "random_noise":
        return random_noise(x, y, augmentation_prob)
    else:
        raise ValueError("The augmentation type must be one of 'cutmix', 'mixup' or 'cutout'")

def preprocess_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    encode_categorical: bool,
    categorical_indicator: List,
    attribute_names: List,
    test_split_size=0.2,
    seed=11,
    encoding_type: str = "ordinal",
) -> Dict:

    dropped_column_names = []
    dropped_column_indices = []

    for column_index, column_name in enumerate(X.keys()):
        if X[column_name].isnull().sum() > len(X[column_name]) * 0.9:
            dropped_column_names.append(column_name)
            dropped_column_indices.append(column_index)

    for column_index, column_name in enumerate(X.keys()):
        if X[column_name].dtype == 'object' or X[column_name].dtype == 'category' or X[column_name].dtype == 'string':
            if X[column_name].nunique() / len(X[column_name]) > 0.9:
                dropped_column_names.append(column_name)
                dropped_column_indices.append(column_index)

    X = X.drop(dropped_column_names, axis=1)
    attribute_names = [attribute_name for attribute_name in attribute_names if attribute_name not in dropped_column_names]
    categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if i not in dropped_column_indices]

    column_category_values = []
    # take pandas categories into account
    for cat_indicator, column_name in zip(categorical_indicator, X.keys()):
        if cat_indicator:
            column_categories = list(X[column_name].cat.categories)
            column_category_values.append(column_categories)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_split_size,
        random_state=seed,
        stratify=y,
    )

    numerical_features = [i for i in range(len(categorical_indicator)) if not categorical_indicator[i]]
    categorical_features = [i for i in range(len(categorical_indicator)) if categorical_indicator[i]]

    column_types = {}
    for column_name in X_train.keys():
        if X_train[column_name].dtype == 'object' or X_train[column_name].dtype == 'category' or X_train[column_name].dtype == 'string':
            column_types[column_name] = 'category'
        elif pd.api.types.is_numeric_dtype(X_train[column_name]):
            column_types[column_name] = 'float64'
        else:
            raise ValueError("The column type must be one of 'object', 'category', 'string', 'int' or 'float'")

    dataset_preprocessors = []
    if len(numerical_features) > 0:
        numerical_preprocessor = ('numerical', StandardScaler(), numerical_features)
        dataset_preprocessors.append(numerical_preprocessor)
    if len(categorical_features) > 0 and encode_categorical:
        if encoding_type == "ordinal":
            categorical_preprocessor = ('low_card_categorical', TargetEncoder(), categorical_features)
        else:
            categorical_preprocessor = ('low_card_categorical', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)

        dataset_preprocessors.append(categorical_preprocessor)

    column_transformer = ColumnTransformer(
        dataset_preprocessors,
        remainder='passthrough',
    )
    X_train = column_transformer.fit_transform(X_train, y_train)
    X_test = column_transformer.transform(X_test)
    # convert to dataframe and detect dtypes
    # dataframe from numpy array
    # create dataframe from numpy array

    if len(numerical_features) > 0:
        new_categorical_indicator = [False] * len(numerical_features)
        new_attribute_names = [attribute_names[i] for i in numerical_features]
    else:
        new_categorical_indicator = []
        new_attribute_names = []

    if len(categorical_features) > 0:
        if encoding_type == "ordinal":
            new_categorical_indicator.extend([True] * len(categorical_features))
            new_attribute_names.extend([attribute_names[i] for i in categorical_features])
        else:
            for i in range(len(column_category_values)):
                if len(column_category_values[i]) > 2:
                    new_categorical_indicator.extend([True] * len(column_category_values[i]))
                    new_attribute_names.extend([attribute_names[categorical_features[i]] + '_' + str(category) for category in column_category_values[i]])
                else:
                    new_categorical_indicator.extend([True])
                    new_attribute_names.extend([attribute_names[categorical_features[i]]])

    X_train = pd.DataFrame(X_train, columns=new_attribute_names)
    X_test = pd.DataFrame(X_test, columns=new_attribute_names)

    if encoding_type == "ordinal":
        X_train = X_train.astype(column_types)
        X_test = X_test.astype(column_types)

    # pandas fill missing values for numerical columns with zeroes
    for cat_indicator, column_name in zip(new_categorical_indicator, X_train.keys()):
        if not cat_indicator:
            X_train[column_name] = X_train[column_name].fillna(0)
            X_test[column_name] = X_test[column_name].fillna(0)
        else:
            # categorical variables where not encoded
            if not encode_categorical:
                X_train[column_name] = X_train[column_name].fillna('missing')
                X_test[column_name] = X_test[column_name].fillna('missing')

    # scikit learn label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    info_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'categorical_indicator': new_categorical_indicator,
        'attribute_names': new_attribute_names,
    }

    return info_dict

def get_dataset(dataset_id: int, test_split_size=0.2, seed=11, encode_categorical: bool = True, encoding_type: str ='ordinal') -> Dict:

    # Get the data
    dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
    dataset_name = dataset.name
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    info_dict = preprocess_dataset(
        X,
        y,
        encode_categorical,
        categorical_indicator,
        attribute_names,
        test_split_size=test_split_size,
        seed=seed,
        encoding_type=encoding_type,
    )
    info_dict['dataset_name'] = dataset_name
    return info_dict
