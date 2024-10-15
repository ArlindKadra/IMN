import argparse
import json
import os
from typing import Dict

import optuna
import numpy as np
import pandas as pd

from baseline_experiment import main
from search_spaces import (
    hpo_space_logistic,
    hpo_space_dtree,
    hpo_space_random_forest,
    hpo_space_catboost,
    hpo_space_tabnet,
)
from utils import get_dataset


ENCODE_CATEGORICAL_VARIABLES = {
    'random_forest': True,
    'catboost': True,
    'decision_tree': True,
    'logistic_regression': True,
    'tabnet': False,
}


def objective(
    trial: optuna.trial.Trial,
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    categorical_indicator: np.ndarray,
    attribute_names: np.ndarray,
    dataset_name: str,
) -> float:
    """The objective function for hyperparameter optimization.

    Args:
        trial: The optuna trial object.
        args: The arguments for the experiment.
        X_train: The training examples.
        y_train: The training labels.
        X_valid: The validation examples.
        y_valid: The validation labels.
        categorical_indicator: The categorical indicator for the features.
        attribute_names: The feature names.
        dataset_name: The name of the dataset.

    Returns:
        The test AUROC.
    """
    if args.model_name == 'logistic_regression':
        hp_config = hpo_space_logistic(trial)
    elif args.model_name == 'decision_tree':
        hp_config = hpo_space_dtree(trial)
    elif args.model_name == 'catboost':
        hp_config = hpo_space_catboost(trial)
    elif args.model_name == 'random_forest':
        hp_config = hpo_space_random_forest(trial)
    elif args.model_name == 'tabnet':
        hp_config = hpo_space_tabnet(trial)

    output_info = main(
        args,
        hp_config,
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_indicator,
        attribute_names,
        dataset_name,
    )

    return output_info['test_auroc']


def hpo_main(args):
    """The main function for hyperparameter optimization."""
    info = get_dataset(
        args.dataset_id,
        test_split_size=args.test_split_size,
        seed=args.seed,
        encode_categorical=ENCODE_CATEGORICAL_VARIABLES[args.model_name],
        hpo_tuning=args.hpo_tuning,

    )

    dataset_name = info['dataset_name']
    attribute_names = info['attribute_names']

    X_train = info['X_train']
    X_test = info['X_test']

    y_train = info['y_train']
    y_test = info['y_test']

    if args.hpo_tuning:
        X_valid = info['X_valid']
        y_valid = info['y_valid']

    categorical_indicator = info['categorical_indicator']

    output_directory = os.path.join(args.output_dir, f'{args.model_name}', f'{args.dataset_id}', f'{args.seed}')
    os.makedirs(output_directory, exist_ok=True)

    if args.hpo_tuning:

        time_limit = 60 * 60
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

        try:
            study.optimize(
                lambda trial: objective(
                    trial,
                    args,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    categorical_indicator,
                    attribute_names,
                    dataset_name,
                ),
                n_trials=args.n_trials,
                timeout=time_limit,
            )
        except Exception as e:
            print(f'Optimization stopped: {e}')

        try:
            best_params = study.best_params
        except ValueError:
            best_params = None

        trial_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        trial_df.to_csv(os.path.join(output_directory, 'trials.csv'), index=False)

    # concatenate train and validation as pandas
    X_train = pd.concat([X_train, X_valid], axis=0)
    y_train = np.concatenate([y_train, y_valid], axis=0)

    output_info = main(
        args,
        best_params if args.hpo_tuning else None,
        X_train,
        y_train,
        X_test,
        y_test,
        categorical_indicator,
        attribute_names,
        dataset_name,
    )

    with open(os.path.join(output_directory, 'output_info.json'), 'w') as f:
        json.dump(output_info, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=1590,
        help='Dataset id'
    )
    parser.add_argument(
        '--test_split_size',
        type=float,
        default=0.2,
        help='Test size'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='tabnet',
        help='The name of the baseline model to use',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the results',
    )
    parser.add_argument(
        '--hpo_tuning',
        action='store_true',
        help='Whether to perform hyperparameter tuning',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='Number of trials for hyperparameter tuning',
    )
    parser.add_argument(
        '--disable_wandb',
        action='store_true',
        help='Whether to disable wandb logging',
    )

    args = parser.parse_args()

    hpo_main(args)
