import argparse
import time
from typing import Dict

import shap
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

import torch


def main(
    args: argparse.Namespace,
    hp_config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    categorical_indicator: np.ndarray,
    attribute_names: np.ndarray,
    dataset_name: str,
) -> Dict:
    """Main entry point for the experiment.

    Args:
        args: The arguments for the experiment.
        hp_config: The hyperparameter configuration.
        X_train: The training examples.
        y_train: The training labels.
        X_test: The test examples.
        y_test: The test labels.
        categorical_indicator: The categorical indicator for the features.
        attribute_names: The feature names.
        dataset_name: The name of the dataset.

    Returns:
        output_info: A dictionary with the main results from the experiment.
    """
    np.random.seed(args.seed)
    seed = args.seed

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    #categorical_indices = [i for i, cat_indicator in enumerate(categorical_indicator) if cat_indicator]
    # count number of unique categories per pandas column
    #categorical_counts = [len(np.unique(X_train.iloc[:, i])) for i in categorical_indices]

    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    nr_classes = len(unique_classes)

    if not args.disable_wandb:
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
       # "cat_dims": categorical_counts if nr_categorical > 0 else [],
        "seed": seed,
        "device_name": "cpu",
        'optimizer_fn': torch.optim.AdamW,
    }

    basic_hp_config_logistic = {
        'random_state': seed,
        'class_weight': 'balanced',
        'multi_class': 'multinomial' if nr_classes > 2 else 'ovr',
    }
    basic_hp_config_dtree = {
        'random_state': seed,
        'class_weight': 'balanced',
    }
    basic_hp_config_catboost = {
        'task_type': 'GPU',
        'devices': '0',
        'loss_function': 'MultiClass' if nr_classes > 2 else 'Logloss',
        'random_state': seed,
        'class_weights': class_weights,
    }

    basic_hp_config_random_forest = {
        'random_state': seed,
        'class_weight': 'balanced',
    }


    if hp_config is not None:
        if args.model_name == 'logistic_regression':
            basic_hp_config_logistic.update(hp_config)
        elif args.model_name == 'decision_tree':
            basic_hp_config_dtree.update(hp_config)
        elif args.model_name == 'catboost':
            basic_hp_config_catboost.update(hp_config)
        elif args.model_name == 'random_forest':
            basic_hp_config_random_forest.update(hp_config)
        elif args.model_name == 'tabnet':
            tabnet_params.update(hp_config)

    if args.model_name == 'random_forest':
        model = RandomForestClassifier(**basic_hp_config_random_forest)
    elif args.model_name == 'catboost':
        model = CatBoostClassifier(
            **basic_hp_config_catboost,
        )
    elif args.model_name == 'decision_tree':
        model = DecisionTreeClassifier(**basic_hp_config_dtree)
    elif args.model_name == 'logistic_regression':
        model = LogisticRegression(**basic_hp_config_logistic)
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
            column_transformer.fit(np.concatenate((X_train, X_test), axis=0))
            X_train = column_transformer.transform(X_train)
            X_test = column_transformer.transform(X_test)
        else:
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

        tabnet_not_default = False
        if 'learning_rate' in tabnet_params:
            tabnet_not_default = True
            optimizer_params = {'lr': tabnet_params['learning_rate']}
            scheduler_params = dict(decay_rate=tabnet_params['decay_rate'], decay_iterations=tabnet_params['decay_iterations'])
            tabnet_params['optimizer_params'] = optimizer_params
            tabnet_params['scheduler_params'] = scheduler_params

            del tabnet_params['learning_rate']
            del tabnet_params['decay_rate']
            del tabnet_params['decay_iterations']

            batch_size = tabnet_params['batch_size']
            virtual_batch_size = tabnet_params['virtual_batch_size']
            epochs = tabnet_params['epochs']
            del tabnet_params['batch_size']
            del tabnet_params['virtual_batch_size']
            del tabnet_params['epochs']

        model = TabNetClassifier(**tabnet_params)


    if args.model_name == 'catboost':
        model.fit(X_train, y_train) #cat_features=categorical_indices)
    elif args.model_name == 'tabnet':
        if tabnet_not_default:
            model.fit(X_train, y_train, weights=1, batch_size=batch_size, virtual_batch_size=virtual_batch_size, max_epochs=epochs, eval_metric=['auc'],)
        else:
            model.fit(X_train, y_train, weights=1, eval_metric=['auc'])
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_time

    predict_start = time.time()

    train_predictions_labels = model.predict(X_train)

    predict_time = time.time() - predict_start
    print("Predict time: %s" % predict_time)
    train_predictions_probabilities = model.predict_proba(X_train)[:, 1] if nr_classes == 2 else model.predict_proba(X_train)
    test_predictions_labels = model.predict(X_test)
    test_predictions_probabilities = model.predict_proba(X_test)
    if nr_classes == 2:
        test_predictions_probabilities = model.predict_proba(X_test)[:, 1]

    start_time = time.time()

    # calculate the balanced accuracy
    train_auroc = roc_auc_score(y_train, train_predictions_probabilities, multi_class='raise' if nr_classes == 2 else 'ovo')
    train_accuracy = accuracy_score(y_train, train_predictions_labels)
    test_auroc = roc_auc_score(y_test, test_predictions_probabilities, multi_class='raise' if nr_classes == 2 else 'ovo')
    test_accuracy = accuracy_score(y_test, test_predictions_labels)

    inference_time = time.time() - train_time - start_time

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

    # get the names of the top features
    top_features = [attribute_names[i] for i in sorted_idx]
    top_importances = [feature_importances[i] for i in sorted_idx]
    print("Top features: %s" % top_features)
    print("Top feature importances: %s" % top_importances)

    if not args.disable_wandb:
        wandb.run.summary["Train:auroc"] = train_auroc
        wandb.run.summary["Train:accuracy"] = train_accuracy
        wandb.run.summary["Test:auroc"] = test_auroc
        wandb.run.summary["Test:accuracy"] = test_accuracy
        wandb.run.summary["Top_features"] = top_features
        wandb.run.summary["Top_features_weights"] = top_importances
        wandb.run.summary["Train:time"] = train_time
        wandb.run.summary["Inference:time"] = inference_time
        wandb.finish()

    output_info = {
        'train_auroc': train_auroc,
        'train_accuracy': train_accuracy,
        'test_auroc': test_auroc,
        'test_accuracy': test_accuracy,
        'top_features': top_features,
        'top_features_weights': top_importances,
        'train_time': train_time,
        'inference_time': inference_time,
    }

    return output_info
