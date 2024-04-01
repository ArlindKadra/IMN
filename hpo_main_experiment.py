import argparse
import json
import os
from typing import Dict

import optuna
import numpy as np
import pandas as pd

from main_experiment import main
from utils import get_dataset


def hpo_space_imn(trial: optuna.trial.Trial) -> Dict:

    params = {
        'nr_epochs': trial.suggest_int('nr_epochs', 10, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'weight_norm': trial.suggest_float('weight_norm', 1e-5, 1e-1, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.5),
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

    if args.interpretable:
        hp_config = hpo_space_imn(trial)

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


    info = get_dataset(
        dataset_id,
        test_split_size=test_split_size,
        seed=seed,
        encode_categorical=True,
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
    model_name = 'inn' if args.interpretable else 'tabresnet'
    output_directory = os.path.join(args.output_dir, model_name, f'{args.dataset_id}', f'{seed}')
    os.makedirs(output_directory, exist_ok=True)

    if args.hpo_tuning:

        time_limit = 60 * 60
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.enqueue_trial(
            {
                'nr_epochs': 500,
                'batch_size': 64,
                'learning_rate': 0.01,
                'weight_decay': 0.01,
                'weight_norm': 0.1,
                'dropout_rate': 0.25,
            }
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
        "--nr_blocks",
        type=int,
        default=2,
        help="Number of levels in the hypernetwork",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Number of hidden units in the hypernetwork",
    )
    parser.add_argument(
        "--augmentation_probability",
        type=float,
        default=0,
        help="Probability of data augmentation",
    )
    parser.add_argument(
        "--scheduler_t_mult",
        type=int,
        default=2,
        help="Multiplier for the scheduler",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed',
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=31,
        help='Dataset id',
    )
    parser.add_argument(
        '--test_split_size',
        type=float,
        default=0.2,
        help='Test size',
    )
    parser.add_argument(
        '--nr_restarts',
        type=int,
        default=3,
        help='Number of learning rate restarts',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the results',
    )
    parser.add_argument(
        '--interpretable',
        action='store_true',
        default=False,
        help='Whether to use interpretable models',
    )
    parser.add_argument(
        '--encoding_type',
        type=str,
        default='ordinal',
        help='Encoding type',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='classification',
        help='If we are doing classification or regression.',
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