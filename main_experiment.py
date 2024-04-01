import argparse
import json
import os
import time
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import torch
import wandb

from models.model import Classifier
from utils import get_dataset


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

    dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_id = args.dataset_id

    if hp_config is None:
        hp_config = {
            'nr_epochs': 500,
            'batch_size': 64,
            'learning_rate': 0.01,
            'weight_decay': 0.01,
            'weight_norm': 0.1,
            'dropout_rate': 0.25,
        }

    seed = args.seed

    X_train = X_train.to_numpy()
    X_train = X_train.astype(np.float32)
    X_test = X_test.to_numpy()
    X_test = X_test.astype(np.float32)

    nr_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    nr_classes = len(unique_classes)

    # separate into classes
    dataset_classes = {}
    for i in range(nr_classes):
        dataset_classes[i] = []

    for index, label in enumerate(y_train):
        dataset_classes[label].append(index)

    majority_class_nr = -1
    for i in range(nr_classes):
        if len(dataset_classes[i]) > majority_class_nr:
            majority_class_nr = len(dataset_classes[i])

    examples_train = []
    labels_train = []

    for i in range(nr_classes):
        nr_instances_class = len(dataset_classes[i])
        if nr_instances_class < majority_class_nr:
            # oversample
            oversampled_indices = np.random.choice(
                dataset_classes[i],
                majority_class_nr - nr_instances_class,
                replace=True,
            )
            examples_train.extend(X_train[dataset_classes[i]])
            labels_train.extend(y_train[dataset_classes[i]])
            for index in oversampled_indices:
                examples_train.append(X_train[index])
                labels_train.append(y_train[index])
        else:
            examples_train.extend(X_train[dataset_classes[i]])
            labels_train.extend(y_train[dataset_classes[i]])

    network_configuration = {
        'nr_features': nr_features,
        'nr_classes': nr_classes if nr_classes > 2 else 1,
        'nr_blocks': args.nr_blocks,
        'hidden_size': args.hidden_size,
        'dropout_rate': hp_config['dropout_rate'],
    }


    interpretable = args.interpretable
    if not args.disable_wandb:
        wandb.init(
            project='INN',
            config=args,
        )
        wandb.config['weight_norm'] = hp_config['weight_norm']
        wandb.config['model_name'] = 'inn' if interpretable else 'tabresnet'
        wandb.config['dataset_name'] = dataset_name

    output_directory = os.path.join(
        args.output_dir,
        model_name,
        f'{dataset_id}',
        f'{seed}',

    )
    os.makedirs(output_directory, exist_ok=True)

    args.nr_epochs = hp_config['nr_epochs']
    args.learning_rate = hp_config['learning_rate']
    args.batch_size = hp_config['batch_size']
    args.weight_decay = hp_config['weight_decay']
    args.weight_norm = hp_config['weight_norm']
    args.dropout_rate = hp_config['dropout_rate']

    model = Classifier(
        network_configuration,
        args=args,
        categorical_indicator=categorical_indicator,
        attribute_names=attribute_names,
        model_name=model_name,
        device=dev,
        output_directory=output_directory,
        disable_wandb=args.disable_wandb,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    if interpretable:
        test_predictions, weight_importances = model.predict(X_test, y_test, return_weights=True)
        train_predictions = model.predict(X_train, y_train)
    else:
        test_predictions = model.predict(X_test, y_test)
        train_predictions = model.predict(X_train, y_test)

    inference_time = time.time() - start_time - train_time

    # from series to list
    y_test = y_test.tolist()
    y_train = y_train.tolist()

    if args.mode == 'classification':

        test_auroc = roc_auc_score(y_test, test_predictions, multi_class='raise' if nr_classes == 2 else 'ovo')
        train_auroc = roc_auc_score(y_train, train_predictions, multi_class='raise' if nr_classes == 2 else 'ovo')

        # threshold the predictions if the model is binary
        if nr_classes == 2:
            test_predictions = (test_predictions > 0.5).astype(int)
            train_predictions = (train_predictions > 0.5).astype(int)
        else:
            test_predictions = np.argmax(test_predictions, axis=1)
            train_predictions = np.argmax(train_predictions, axis=1)

        test_accuracy = accuracy_score(y_test, test_predictions)
        train_accuracy = accuracy_score(y_train, train_predictions)
        if not args.disable_wandb:
            wandb.run.summary["Test:accuracy"] = test_accuracy
            wandb.run.summary["Test:auroc"] = test_auroc
            wandb.run.summary["Train:accuracy"] = train_accuracy
            wandb.run.summary["Train:auroc"] = train_auroc
    else:
        test_mse = mean_squared_error(y_test, test_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        if not args.disable_wandb:
            wandb.run.summary["Test:mse"] = test_mse
            wandb.run.summary["Train:mse"] = train_mse

    if args.mode == 'classification':
        output_info = {
            'train_auroc': train_auroc,
            'train_accuracy': train_accuracy,
            'test_auroc': test_auroc,
            'test_accuracy': test_accuracy,
            'train_time': train_time,
            'inference_time': inference_time,
        }
    else:
        output_info = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_time': train_time,
            'inference_time': inference_time,
        }

    if interpretable:
        # print attribute name and weight for the top 10 features
        # average the weight_importances
        weight_importances = np.mean(weight_importances, axis=0)
        #weight_importances = weight_importances[:-1]
        sorted_idx = np.argsort(weight_importances)[::-1]
        top_10_features = [attribute_names[i] for i in sorted_idx]
        print("Top 10 features: %s" % top_10_features)
        # print the weights of the top 10 features
        print(weight_importances[sorted_idx])
        output_info['top_10_features'] = top_10_features
        output_info['top_10_features_weights'] = weight_importances[sorted_idx].tolist()
        if not args.disable_wandb:
            wandb.run.summary["Top_10_features"] = top_10_features
            wandb.run.summary["Top_10_features_weights"] = weight_importances[sorted_idx]

    if not args.disable_wandb:
        wandb.finish()

    return output_info
