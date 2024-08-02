import argparse
import time
from typing import Dict

#import shap
from catboost import CatBoostClassifier
from DAN_Task import DANetClassifier, DANetRegressor
from hypertab import HyperTabClassifier
from qhoptim.pyt import QHAdam
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
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

    np.random.seed(args.seed)

    seed = args.seed


    categorical_indices = [i for i, cat_indicator in enumerate(categorical_indicator) if cat_indicator]
    # count number of unique categories per pandas column
    categorical_counts = [len(np.unique(X_train.iloc[:, i])) for i in categorical_indices]

    X_train = np.array(X_train)
    X_test = np.array(X_test)

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
        "cat_dims": categorical_counts if nr_categorical > 0 else [],
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

        tabnet_not_default = False
        if 'learning_rate' in tabnet_params:
            tabnet_not_default = True
            optimizer_params = {'lr': tabnet_params['learning_rate']}
            scheduler_params = dict(decay_rate=tabnet_params['decay_rate'],
                                        decay_iterations=tabnet_params['decay_iterations'])
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
    elif args.model_name == 'danet':
        model = DANetClassifier(
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=0.008, weight_decay=1e-5, nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=20),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=20,
            base_outdim=64,
            k=5,
            drop_rate=0.1,
            seed=seed,
        )
    elif args.model_name == 'hypertab':
        model = HyperTabClassifier(device='cuda', epochs=50)
    else:
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()


    if args.model_name == 'catboost':
        model.fit(X_train, y_train, cat_features=categorical_indices)
    elif args.model_name == 'tabnet':
        if tabnet_not_default:
            model.fit(X_train, y_train, weights=1, batch_size=batch_size, virtual_batch_size=virtual_batch_size, max_epochs=epochs, eval_metric=['auc'],)
        else:
            model.fit(X_train, y_train, weights=1, eval_metric=['auc'])
    elif args.model_name == 'danet':
            model.fit(
                X_train=X_train,
                y_train=y_train,
                max_epochs=500,
                patience=50,
                batch_size=8192,
                virtual_batch_size=256,
            )
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_time

    predict_start = time.time()

    train_predictions_labels = model.predict(X_train)
    predict_time = time.time() - predict_start
    print("Predict time: %s" % predict_time)
    train_predictions_probabilities = model.predict_proba(X_train)[:, 1] if nr_classes == 2 else model.predict_proba(X_train)
    test_predictions_labels = model.predict(X_test)
    test_predictions_probabilities = model.predict_proba(X_test)[:, 1] if nr_classes == 2 else model.predict_proba(X_test)

    """
    start_time = time.time()
    def f(X):
        return model.predict(X)

    med = np.median(X_test, axis=0).reshape((1, X_test.shape[1]))
    print(med.shape)

    explainer = shap.Explainer(f, med)
    shap_weights = []
    # reshape example

    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()

    for i in range(X_test.shape[0]):
        example = X_test[i, :]

        example = example.reshape((1, X_test.shape[1]))
        shap_values = explainer.shap_values(example)
        shap_weights.append(shap_values)
    shap_weights = np.array(shap_weights)
    shap_weights = np.squeeze(shap_weights, axis=1)
    shap_weights = np.mean(np.abs(shap_weights), axis=0)
    shap_weights = shap_weights / np.sum(shap_weights)
    print(shap_weights)
    end_time = time.time()
    print(f"SHAP time: {end_time - start_time}")
    """
    # calculate the balanced accuracy
    train_auroc = roc_auc_score(y_train, train_predictions_probabilities) if nr_classes == 2 else roc_auc_score(y_train, train_predictions_probabilities, multi_class='ovo')
    train_accuracy = accuracy_score(y_train, train_predictions_labels)
    test_auroc = roc_auc_score(y_test, test_predictions_probabilities) if nr_classes == 2 else roc_auc_score(y_test, test_predictions_probabilities, multi_class='ovo')
    test_accuracy = accuracy_score(y_test, test_predictions_labels)

    inference_time = time.time() - train_time - start_time
    if args.model_name != 'danet' and args.model_name != 'hypertab':
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

    if not args.disable_wandb:
        wandb.run.summary["Train:auroc"] = train_auroc
        wandb.run.summary["Train:accuracy"] = train_accuracy
        wandb.run.summary["Test:auroc"] = test_auroc
        wandb.run.summary["Test:accuracy"] = test_accuracy
        wandb.run.summary["Train:time"] = train_time
        wandb.run.summary["Inference:time"] = inference_time
        if args.model_name != 'danet' and args.model_name != 'hypertab':
            wandb.run.summary["Top_10_features"] = top_10_features
            wandb.run.summary["Top_10_features_weights"] = top_10_importances

        wandb.finish()

    output_info = {
        'train_auroc': train_auroc,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_auroc': test_auroc,
        'train_time': train_time,
        'inference_time': inference_time,
    }

    if args.model_name != 'danet' and args.model_name != 'hypertab':
        output_info['top_10_features'] = top_10_features
        output_info['top_10_importances'] = top_10_importances

    return output_info