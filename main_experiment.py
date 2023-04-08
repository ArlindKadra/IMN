import argparse
from copy import deepcopy
import json
import os
import time

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR
from torcheval.metrics.functional import binary_auroc, multiclass_auroc, binary_accuracy, multiclass_accuracy
import torch
import pandas as pd
import numpy as np
import wandb

from models.hypernetwork import HyperNet
from utils import augment_data, get_dataset


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
    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    augmentation_probability = args.augmentation_probability
    weight_decay = args.weight_decay
    scheduler_t_mult = args.scheduler_t_mult
    nr_restarts = args.nr_restarts

    info = get_dataset(
        dataset_id,
        test_split_size=test_split_size,
        seed=seed,
        encoding_type=args.encoding_type,
    )

    dataset_name = info['dataset_name']
    X_train = info['X_train'].to_numpy()
    X_train = X_train.astype(np.float32)
    X_test = info['X_test'].to_numpy()
    X_test = X_test.astype(np.float32)
    y_train = info['y_train']
    y_test = info['y_test']
    categorical_indicator = info['categorical_indicator']
    attribute_names = info['attribute_names']

    # the reference to info is not needed anymore
    del info

    numerical_features = [i for i in range(len(categorical_indicator)) if not categorical_indicator[i]]
    nr_features = X_train.shape[1]
    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    nr_classes = len(unique_classes)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    instance_weights = [class_weights[y_train[i]] for i in range(y_train.shape[0])]

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
    interpretable = args.interpretable
    wandb.config['model_name'] = 'inn' if interpretable else 'tabresnet'

    start_time = time.time()
    # Train a hypernetwork
    hypernet = HyperNet(**network_configuration)
    hypernet = hypernet.to(dev)
    try:
        hypernet = torch.compile(hypernet)
    except RuntimeError:
        print('Could not compile the hypernetwork. Continuing without compilation.')

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float() if nr_classes == 2 else torch.tensor(y_train).long()
    X_train = X_train.to(dev)
    y_train = y_train.to(dev)
    # Create dataloader for training
    train_dataset = torch.utils.data.TensorDataset(
        X_train,
        y_train,
    )

    sampler = torch.utils.data.sampler.WeightedRandomSampler(instance_weights, X_train.size(0), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    T_0: int = max(((nr_epochs * len(train_loader)) * (scheduler_t_mult - 1)) // (scheduler_t_mult ** nr_restarts - 1), 1)
    # Train the hypernetwork
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0, scheduler_t_mult)

    def warmup(current_step: int):
        return float(current_step / (5 * len(train_loader)))

    scheduler1 = LambdaLR(optimizer, lr_lambda=warmup)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5 * len(train_loader)])
    if nr_classes > 2:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    wandb.watch(hypernet, criterion, log='all', log_freq=10)
    ensemble_snapshot_intervals = [T_0, (scheduler_t_mult + 1) * T_0, (scheduler_t_mult ** 2 + scheduler_t_mult + 1) * T_0]

    ensemble_snapshots = []
    iteration = 0

    sigmoid_act_func = torch.nn.Sigmoid()
    softmax_act_func = torch.nn.Softmax(dim=1)
    loss_per_epoch = []
    weight_norm = 0.1 / (batch_size * (nr_classes if nr_classes > 2 else 1))
    train_auroc_per_epoch = []
    for epoch in range(1, nr_epochs + 1):

        loss_value = 0
        train_auroc = 0
        for batch_idx, batch in enumerate(train_loader):

            iteration += 1
            x, y = batch
            hypernet.eval()
            info = augment_data(
                x,
                y,
                numerical_features,
                hypernet,
                criterion,
                augmentation_prob=augmentation_probability,
            )
            hypernet.train()
            optimizer.zero_grad()

            if len(info) == 4:
                x, y_1, y_2, lam = info
                if interpretable:
                    output, weights = hypernet(x, return_weights=True)
                else:
                    output = hypernet(x)

                if nr_classes == 2:
                    output = output.squeeze(1)
                main_loss = lam * criterion(output, y_1) + (1 - lam) * criterion(output, y_2)
            else:
                x, adversarial_x, y_1, y_2, lam = info
                if interpretable:
                    output, weights = hypernet(x, return_weights=True)
                    output_adv = hypernet(adversarial_x)
                else:
                    output = hypernet(x)
                    output_adv = hypernet(adversarial_x)

                if nr_classes == 2:
                    output = output.squeeze(1)
                    output_adv = output_adv.squeeze(1)
                main_loss = lam * criterion(output, y_1) + (1 - lam) * criterion(output_adv, y_2)

            if interpretable:
                """
                weights = torch.abs(weights)
                if nr_classes > 2:
                    for train_example_idx in range(weights.shape[0]):
                        correct_class = y[train_example_idx]
                        for predicted_class in range(weights.shape[2]):
                            if predicted_class != correct_class:
                                weights[train_example_idx, :, predicted_class] = 0

                    weights = torch.mean(weights, dim=2)

                weights = torch.mean(weights, dim=0)
                """

                # take all values except the last one (bias)
                if nr_classes > 2:
                    weights = torch.squeeze(weights)
                else:
                    weights = torch.squeeze(weights, dim=2)

                l1_loss = torch.norm(weights, 1)
                """
                if specify_weight_norm:
                    
                    while (weight_norm * l1_loss) > main_loss:
                        weight_norm = weight_norm / 10
                        print(f'Weight norm: {weight_norm}')
                        specify_weight_norm = False
                """
                loss = main_loss + (weight_norm * l1_loss)
            else:
                loss = main_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            # threshold the predictions if the model is binary
            if nr_classes == 2:
                predictions = (sigmoid_act_func(output) > 0.5).int()
            else:
                predictions = torch.argmax(output, dim=1)

            # calculate balanced accuracy with pytorch
            if nr_classes == 2:
                batch_auroc = binary_auroc(output, y)
            else:
                batch_auroc = multiclass_auroc(output, y, num_classes=nr_classes)

            loss_value += loss.item()
            train_auroc += batch_auroc
            if iteration in ensemble_snapshot_intervals:
                ensemble_snapshots.append(deepcopy(hypernet.state_dict()))

        loss_value /= len(train_loader)
        train_auroc /= len(train_loader)
        print(f'Epoch: {epoch}, Loss: {loss_value}, AUROC: {train_auroc}')
        loss_per_epoch.append(loss_value)
        train_auroc_per_epoch.append(train_auroc.detach().to('cpu').item())

        wandb.log({"Train:loss": loss_value, "Train:auroc": train_auroc, "Learning rate": optimizer.param_groups[0]['lr']})

    snapshot_models = []
    for snapshot_idx, snapshot in enumerate(ensemble_snapshots):

        hypernet = HyperNet(**network_configuration)
        hypernet = hypernet.to(dev)
        hypernet.load_state_dict(snapshot)
        hypernet.eval()
        snapshot_models.append(hypernet)

    X_test = torch.tensor(X_test).float()
    X_test = X_test.to(dev)
    predictions = []
    weights = []
    for snapshot_idx, snapshot in enumerate(snapshot_models):
        with torch.no_grad():
            if interpretable:
                output, model_weights = snapshot(X_test, return_weights=True)
            else:
                output = snapshot(X_test)
            output = output.squeeze(1)
            if nr_classes > 2:
                output = softmax_act_func(output)
            else:
                output = sigmoid_act_func(output)
            predictions.append([output.detach().to('cpu').numpy()])
            if interpretable:
                weights.append([np.abs(model_weights.detach().to('cpu').numpy())])

    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    predictions = np.squeeze(predictions)

    # from series to list
    y_test = y_test.tolist()
    # threshold the predictions if the model is binary

    if nr_classes == 2:
        test_auroc = roc_auc_score(y_test, predictions)
    else:
        test_auroc = roc_auc_score(y_test, predictions, multi_class="ovo")

    if nr_classes == 2:
        predictions = (predictions > 0.5).astype(int)
    else:
        predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    print("AUROC: %0.3f" % test_auroc)
    print("Accuracy: %0.3f" % accuracy)

    wandb.run.summary["Test:accuracy"] = accuracy
    wandb.run.summary["Test:auroc"] = test_auroc
    if interpretable:
        weights = np.array(weights)
        weights = np.squeeze(weights)
        if len(weights.shape) > 2:
            weights = weights[-1, :, :]
            weights = np.squeeze(weights)

        selected_weights = []
        for test_example_idx in range(weights.shape[0]):
            # select the weights for the predicted class
            if y_test[test_example_idx] == predictions[test_example_idx]:
                if nr_classes > 2:
                    selected_weights.append(weights[test_example_idx, :, predictions[test_example_idx]])
                else:
                    selected_weights.append(weights[test_example_idx, :])

        weights = np.array(selected_weights)
        weights = weights[:, :-1]

        # sum the weights over all test examples
        weights = np.sum(weights, axis=0)

        # normalize the weights
        weights = weights / np.sum(weights)

        # print attribute name and weight for the top 10 features
        sorted_idx = np.argsort(weights)[::-1]
        top_10_features = [attribute_names[i] for i in sorted_idx[:10]]
        print("Top 10 features: %s" % top_10_features)
        # print the weights of the top 10 features
        print(weights[sorted_idx[:10]])
        wandb.run.summary["Top_10_features"] = top_10_features
        wandb.run.summary["Top_10_features_weights"] = weights[sorted_idx[:10]]

    end_time = time.time()
    output_info = {
        'train_auroc': train_auroc_per_epoch,
        'train_loss': loss_per_epoch,
        'test_accuracy': accuracy,
        'test_auroc': test_auroc,
        'time': end_time - start_time,
    }

    if interpretable:
        output_info['top_10_features'] = top_10_features
        output_info['top_10_features_weights'] = weights[sorted_idx[:10]].tolist()

    output_directory = os.path.join(args.output_dir, 'inn' if interpretable else 'tabresnet', f'{dataset_id}', f'{seed}')
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, 'output_info.json'), 'w') as f:
        json.dump(output_info, f)

    # save model
    torch.save(hypernet.state_dict(), os.path.join(output_directory, 'model.pt'))

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
        default=0.2,
        help="Probability of data augmentation",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
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
        default=11,
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
        default=True,
        help='Whether to use interpretable models',
    )
    parser.add_argument(
        '--encoding_type',
        type=str,
        default='ordinal',
        help='Encoding type',
    )

    args = parser.parse_args()

    main(args)
