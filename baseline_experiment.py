import argparse
import json
import os
import time

from catboost import CatBoostClassifier
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight
import wandb

from utils import get_dataset


def main(args: argparse.Namespace):

    np.random.seed(args.seed)

    dataset_id = args.dataset_id
    test_split_size = args.test_split_size
    seed = args.seed

    encode_categorical_variables = {
        'catboost': False,
        'decision_tree': True,
        'logistic_regression': True,
        'random_forest': True,
        'tabnet': False,
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
    # count number of unique categories per pandas column
    categorical_counts = [len(np.unique(X_train.iloc[:, i])) for i in categorical_indices]
    attribute_names = info['attribute_names']

    # the reference to info is not needed anymore
    del info

    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    nr_classes = len(unique_classes)
    wandb.init(
        project='INN',
        config=args,
    )
    wandb.config['dataset_name'] = dataset_name
    start_time = time.time()
    # count number of categorical variables
    nr_categorical = np.sum(categorical_indicator)
    tabnet_params = {
        "cat_idxs": [i for i in range(nr_categorical)] if nr_categorical > 0 else [],
        "cat_dims": categorical_counts if nr_categorical > 0 else [],
        "seed": seed,
        "device_name": "cuda",
    }

    if args.model_name == 'random_forest':
        model = RandomForestClassifier(random_state=seed, class_weight='balanced')
    elif args.model_name == 'catboost':
        model = CatBoostClassifier(
            task_type='GPU',
            devices='0',
            loss_function='MultiClass' if nr_classes > 2 else 'Logloss',
            class_weights=class_weights,
            random_seed=seed,
        )
    elif args.model_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=seed)
    elif args.model_name == 'logistic_regression':
        model = LogisticRegression(random_state=seed, class_weight='balanced', multi_class='multinomial' if nr_classes > 2 else 'ovr')
    elif args.model_name == 'tabnet':
        if nr_categorical > 0:
            cat_attribute_names = [attribute_names[i] for i in categorical_indices]
            numerical_attribute_names = [attribute_names[i] for i in range(len(attribute_names)) if i not in categorical_indices]
            attribute_names = cat_attribute_names
            attribute_names.extend(numerical_attribute_names)
            categorical_preprocessor = (
                'categorical_encoder',
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_indicator,
            )
            column_transformer = ColumnTransformer(
                [categorical_preprocessor],
                remainder='passthrough',
            )
            X_train = column_transformer.fit_transform(X_train)
            X_test = column_transformer.transform(X_test)
        else:
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

        model = TabNetClassifier(**tabnet_params)

    if args.model_name == 'catboost':
        model.fit(X_train, y_train, cat_features=categorical_indices)
    elif args.model_name == 'tabnet':
        model.fit(X_train, y_train, weights=1)
    else:
        model.fit(X_train, y_train)

    train_predictions_labels = model.predict(X_train)
    train_predictions_probabilities = model.predict_proba(X_train)
    if nr_classes == 2:
        train_predictions_probabilities = model.predict_proba(X_train)[:, 1]
    test_predictions_labels = model.predict(X_test)
    test_predictions_probabilities = model.predict_proba(X_test)
    if nr_classes == 2:
        test_predictions_probabilities = model.predict_proba(X_test)[:, 1]

    # calculate the balanced accuracy
    train_auroc = roc_auc_score(y_train, train_predictions_probabilities, multi_class='raise' if nr_classes == 2 else 'ovo')
    train_accuracy = accuracy_score(y_train, train_predictions_labels)
    test_auroc = roc_auc_score(y_test, test_predictions_probabilities, multi_class='raise' if nr_classes == 2 else 'ovo')
    test_accuracy = accuracy_score(y_test, test_predictions_labels)

    if args.model_name == 'logistic_regression':
        # get the feature importances
        feature_importances = model.coef_
        if nr_classes > 2:
            feature_importances = np.mean(np.abs(feature_importances), axis=0)
        feature_importances = np.squeeze(feature_importances)
        feature_importances = feature_importances / np.sum(feature_importances)

    else:
        # get the feature importances
        feature_importances = model.feature_importances_

    # sort the feature importances in descending order
    sorted_idx = np.argsort(feature_importances)[::-1]

    if type(feature_importances) == np.ndarray:
        feature_importances = feature_importances.tolist()
    if type(sorted_idx) == np.ndarray:
        sorted_idx = sorted_idx.tolist()

    # get the names of the top 10 features
    top_10_features = [attribute_names[i] for i in sorted_idx]
    top_10_importances = [feature_importances[i] for i in sorted_idx]
    print("Top 10 features: %s" % top_10_features)
    print("Top 10 feature importances: %s" % top_10_importances)

    wandb.run.summary["Train:auroc"] = train_auroc
    wandb.run.summary["Train:accuracy"] = train_accuracy
    wandb.run.summary["Test:auroc"] = test_auroc
    wandb.run.summary["Test:accuracy"] = test_accuracy
    wandb.run.summary["Top_10_features"] = top_10_features
    wandb.run.summary["Top_10_features_weights"] = top_10_importances

    end_time = time.time()
    output_info = {
        'train_auroc': train_auroc,
        'train_accuracy': train_accuracy,
        'test_auroc': test_auroc,
        'test_accuracy': test_accuracy,
        'top_10_features': top_10_features,
        'top_10_features_weights': top_10_importances,
        'time': end_time - start_time,
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

    args = parser.parse_args()

    main(args)
