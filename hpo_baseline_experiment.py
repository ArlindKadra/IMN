import argparse
import json
import os
from typing import Dict

import optuna
import numpy as np
import pandas as pd

from baseline_experiment import main
from utils import get_dataset


def hpo_space_logistic(trial: optuna.trial.Trial) -> Dict:

    params = {
        'C': trial.suggest_float('C', 1e-5, 5),
        'penalty': trial.suggest_categorical('penalty', ['l2', 'none']),
        'max_iter': trial.suggest_int('max_iter', 50, 500),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }

    return params

def hpo_space_dtree(trial: optuna.trial.Trial) -> Dict:

    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 21),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 11),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 3, 26),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
    }

    return params

def hpo_space_catboost(trial: optuna.trial.Trial) -> Dict:

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
        'random_strength': trial.suggest_int('random_strength', 1, 20),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-6, 1, log=True),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 20),
        'iterations': trial.suggest_int('iterations', 100, 4000)

    }

    return params

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

    if args.model_name == 'logistic_regression':
        hp_config = hpo_space_logistic(trial)
    elif args.model_name == 'decision_tree':
        hp_config = hpo_space_dtree(trial)
    elif args.model_name == 'catboost':
        hp_config = hpo_space_catboost(trial)

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

    dataset_id = args.dataset_id
    test_split_size = args.test_split_size
    seed = args.seed

    encode_categorical_variables = {
        'random_forest': True,
        'catboost': False,
        'decision_tree': True,
        'logistic_regression': True,
        'tabnet': False,
    }

    info = get_dataset(
        dataset_id,
        test_split_size=test_split_size,
        seed=seed,
        encode_categorical=encode_categorical_variables[args.model_name],
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

    output_directory = os.path.join(args.output_dir, f'{args.model_name}', f'{args.dataset_id}', f'{seed}')
    os.makedirs(output_directory, exist_ok=True)

    if args.hpo_tuning:

        time_limit = 60 * 60
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        try:
            study.optimize(
                lambda trial: objective(trial, args, X_train, y_train, X_valid, y_valid, categorical_indicator, attribute_names, dataset_name), n_trials=args.n_trials, timeout=time_limit
            )
        except optuna.exceptions.OptunaError as e:
            print(f"Optimization stopped: {e}")

        best_params = study.best_params
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
        default=31,
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
