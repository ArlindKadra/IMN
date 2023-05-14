import argparse
import json
import os
import time

import torch
import numpy as np
import wandb

from models.model import Classifier
from utils import get_dataset
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, mean_squared_error

def main(args: argparse.Namespace) -> None:

    dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_id = args.dataset_id
    test_split_size = args.test_split_size
    seed = args.seed

    info = get_dataset(
        dataset_id,
        test_split_size=test_split_size,
        seed=args.seed,
        encoding_type=args.encoding_type,
    )

    X_train = info['X_train']
    X_test = info['X_test']
    dataset_name = info['dataset_name']

    X_train = X_train.to_numpy()
    X_train = X_train.astype(np.float32)
    X_test = X_test.to_numpy()
    X_test = X_test.astype(np.float32)
    y_train = info['y_train']
    y_test = info['y_test']

    attribute_names = info['attribute_names']
    categorical_indicator = info['categorical_indicator']
    # the reference to info is not needed anymore
    del info

    nr_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    nr_classes = len(unique_classes)

    network_configuration = {
        'nr_features': nr_features,
        'nr_classes': nr_classes if nr_classes > 2 else 1,
        'nr_blocks': args.nr_blocks,
        'hidden_size': args.hidden_size,
    }

    wandb.init(
        project='INN',
        config=args,
    )
    wandb.config['dataset_name'] = dataset_name
    weight_norm = args.weight_norm
    wandb.config['weight_norm'] = weight_norm
    interpretable = args.interpretable
    wandb.config['model_name'] = 'inn' if interpretable else 'tabresnet'

    output_directory = os.path.join(
        args.output_dir,
        'inn' if interpretable else 'tabresnet',
        f'{dataset_id}',
        f'{seed}',

    )
    os.makedirs(output_directory, exist_ok=True)

    start_time = time.time()

    model = Classifier(
        network_configuration,
        args=args,
        categorical_indicator=categorical_indicator,
        attribute_names=attribute_names,
        model_name='inn' if interpretable else 'tabresnet',
        device=dev,
        output_directory=output_directory,
    )

    wandb.config['dataset_name'] = dataset_name

    model.fit(X_train, y_train)
    if interpretable:
        test_predictions, weight_importances = model.predict(X_test, y_test)
        train_predictions, _ = model.predict(X_train, y_train)
    else:
        test_predictions = model.predict(X_test, y_test)
        train_predictions = model.predict(X_train, y_test)

    # from series to list
    y_test = y_test.tolist()
    y_train = y_train.tolist()
    # threshold the predictions if the model is binary

    if args.mode == 'classification':
        if nr_classes == 2:
            test_auroc = roc_auc_score(y_test, test_predictions)
            train_auroc = roc_auc_score(y_train, train_predictions)
        else:
            test_auroc = roc_auc_score(y_test, test_predictions, multi_class="ovo")
            train_auroc = roc_auc_score(y_train, train_predictions, multi_class="ovo")

        if nr_classes == 2:
            test_predictions = (test_predictions > 0.5).astype(int)
            train_predictions = (train_predictions > 0.5).astype(int)
        else:
            test_predictions = np.argmax(test_predictions, axis=1)
            train_predictions = np.argmax(train_predictions, axis=1)


        test_accuracy = accuracy_score(y_test, test_predictions)
        train_accuracy = accuracy_score(y_train, train_predictions)
        wandb.run.summary["Test:accuracy"] = test_accuracy
        wandb.run.summary["Test:auroc"] = test_auroc
        wandb.run.summary["Train:accuracy"] = train_accuracy
        wandb.run.summary["Train:auroc"] = train_auroc
    else:
        test_mse = mean_squared_error(y_test, test_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        wandb.run.summary["Test:mse"] = test_mse
        wandb.run.summary["Train:mse"] = train_mse

    end_time = time.time()
    if args.mode == 'classification':
        output_info = {
            'train_auroc': train_auroc,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_auroc': test_auroc,
            'time': end_time - start_time,
        }
    else:
        output_info = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'time': end_time - start_time,
        }

    """
    def f(X):
        return model.predict([X[:, i] for i in range(X.shape[1])]).flatten()

    med = np.median(X_test, axis=0).reshape((1, X_test.shape[1]))
    explainer = shap.Explainer(f, med)
    shap_weights = []
    # reshape example
    for i in range(X_test.shape[0]):
        example = X_test[i, :]
        example = example.reshape((1, X_test.shape[1]))
        shap_values = explainer.shap_values(example)
        shap_weights.append(shap_values)
    shap_weights = np.array(shap_weights)
    shap_weights = np.squeeze(shap_weights, axis=1)
    shap_weights = np.mean(np.abs(shap_weights), axis=0)
    shap_weights = shap_weights / np.sum(shap_weights)
    """
    if interpretable:
        # print attribute name and weight for the top 10 features
        sorted_idx = np.argsort(weight_importances)[::-1]
        top_10_features = [attribute_names[i] for i in sorted_idx[:10]]
        print("Top 10 features: %s" % top_10_features)
        # print the weights of the top 10 features
        print(weight_importances[sorted_idx[:10]])
        wandb.run.summary["Top_10_features"] = top_10_features
        wandb.run.summary["Top_10_features_weights"] = weight_importances[sorted_idx[:10]]
        output_info['top_10_features'] = top_10_features
        output_info['top_10_features_weights'] = weight_importances[sorted_idx[:10]].tolist()

    with open(os.path.join(output_directory, 'output_info.json'), 'w') as f:
        json.dump(output_info, f)

    wandb.finish()


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
        "--nr_epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--augmentation_probability",
        type=float,
        default=0,
        help="Probability of data augmentation",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--weight_norm",
        type=float,
        default=1,
        help="Weight decay",
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
        default=1590,
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

    args = parser.parse_args()

    main(args)
