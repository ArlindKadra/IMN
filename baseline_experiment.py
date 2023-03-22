import argparse
import json
import os

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import numpy as np
import wandb

from utils import get_dataset


def main(args: argparse.Namespace) -> None:

    np.random.seed(args.seed)

    dataset_id = args.dataset_id
    test_split_size = args.test_split_size
    seed = args.seed

    encode_categorical_variables = {
        'random_forest': True,
        'catboost': False,
    }

    info = get_dataset(
        dataset_id,
        test_split_size=test_split_size,
        seed=seed,
        encode_categorical=encode_categorical_variables[args.model_name],
    )

    dataset_name = info['dataset_name']
    X_train = info['X_train']
    X_test = info['X_test']
    y_train = info['y_train']
    y_test = info['y_test']
    categorical_indicator = info['categorical_indicator']
    categorical_indices = [i for i, cat_indicator in enumerate(categorical_indicator) if cat_indicator]
    attribute_names = info['attribute_names']

    # the reference to info is not needed anymore
    del info

    total_weight = y_train.shape[0]
    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    weight_per_class = total_weight / unique_classes.shape[0]
    weights = (np.ones(unique_classes.shape[0]) * weight_per_class) / class_counts

    wandb.init(
        project='INN',
        config=args,
    )
    wandb.config['dataset_name'] = dataset_name

    if args.model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
    elif args.model_name == 'catboost':
        model = CatBoostClassifier(
            task_type='GPU',
            devices='0',
            loss_function='MultiClass' if len(unique_classes) > 2 else 'Logloss',
            class_weights=weights,
            random_seed=seed,
        )

    if args.model_name == 'catboost':
        model.fit(X_train, y_train, cat_features=categorical_indices)
    else:
        model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # calculate the balanced accuracy
    train_balanced_accuracy = balanced_accuracy_score(y_train, train_predictions)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # get random forest feature importances
    feature_importances = model.feature_importances_
    # sort the feature importances in descending order
    sorted_idx = np.argsort(feature_importances)[::-1]

    if type(feature_importances) == np.ndarray:
        feature_importances = feature_importances.tolist()
    if type(sorted_idx) == np.ndarray:
        sorted_idx = sorted_idx.tolist()

    # get the names of the top 10 features
    top_10_features = [attribute_names[i] for i in sorted_idx[:10]]
    top_10_importances = [feature_importances[i] for i in sorted_idx[:10]]
    print("Top 10 features: %s" % top_10_features)
    print("Top 10 feature importances: %s" % top_10_importances)

    wandb.run.summary["Train:balanced_accuracy"] = train_balanced_accuracy
    wandb.run.summary["Train:accuracy"] = train_accuracy
    wandb.run.summary["Test:balanced_accuracy"] = test_balanced_accuracy
    wandb.run.summary["Test:accuracy"] = test_accuracy
    wandb.run.summary["Top_10_features"] = top_10_features
    wandb.run.summary["Top_10_features_weights"] = top_10_importances

    output_info = {
        'train_balanced_accuracy': train_balanced_accuracy,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_balanced_accuracy': test_balanced_accuracy,
        'top_10_features': top_10_features,
        'top_10_features_weights': top_10_importances,
    }

    output_directory = os.path.join(args.output_dir, f'{args.model_name}', f'{dataset_id}', f'{seed}')
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, 'output_info.json'), 'w') as f:
        json.dump(output_info, f)

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--seed',
        type=int,
        default=11,
        help='Random seed'
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=1111,
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
        default='catboost',
        help='The name of the baseline model to use',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the results',
    )

    args = parser.parse_args()

    main(args)
