import argparse
from copy import deepcopy
import os
from typing import Dict, List, Tuple, Union

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR
from torcheval.metrics.functional import binary_auroc, multiclass_auroc, binary_accuracy, multiclass_accuracy
import numpy as np
import pandas as pd
import wandb

from models.hypernetwork import HyperNet
from models.tabresnet import TabResNet
from utils import augment_data


class Classifier:

    def __init__(
        self,
        network_configuration: Dict,
        args: argparse.Namespace,
        categorical_indicator: List[bool],
        attribute_names: List[str],
        model_name: str = 'inn',
        device: str = 'cpu',
        output_directory: str = '.',
        disable_wandb: bool = True,
    ):
        """Initialize the classifier.

        Args:
            network_configuration: dict
                The configuration for the neural network.
            args: argparse.Namespace
                The arguments controlling the experiment.
            categorical_indicator: list
                A list of booleans indicating whether the corresponding feature is categorical or not.
            attribute_names: list
                A list of strings containing the names of the features.
            model_name: str
                The name of the model to use.
            device: str
                The device to use for training.
            output_directory: str
                The directory to save the results.
            disable_wandb: bool
                Whether to disable wandb logging.
        """
        super(Classifier, self).__init__()

        self.disable_wandb = disable_wandb
        algorithm_backbone = {
            'tabresnet': TabResNet,
            'inn': HyperNet,
        }
        self.nr_classes = network_configuration['nr_classes'] if network_configuration['nr_classes'] != 1 else 2
        if model_name == 'inn':
            self.interpretable = True
        else:
            self.interpretable = False

        self.model = algorithm_backbone[model_name](**network_configuration)
        self.model = self.model.to(device)
        self.args = args
        self.dev = device
        self.mode = args.mode
        self.numerical_features = [i for i in range(len(categorical_indicator)) if not categorical_indicator[i]]
        self.attribute_names = attribute_names
        self.ensemble_snapshots = []
        self.sigmoid_act_func = torch.nn.Sigmoid()
        self.softmax_act_func = torch.nn.Softmax(dim=1)
        self.output_directory = output_directory

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.DataFrame],
    ):
        """Fit the classifier to the data.
        Args:
            X: list, np.ndarray, pd.DataFrame
                The input data.
            y: list, np.ndarray, pd.DataFrame
                The target data.

        Returns:
            self: Classifier
                The fitted classifier.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()
        elif isinstance(y, list):
            y = np.array(y)

        nr_epochs = self.args.nr_epochs
        batch_size = self.args.batch_size
        learning_rate = self.args.learning_rate
        augmentation_probability = self.args.augmentation_probability
        weight_decay = self.args.weight_decay
        scheduler_t_mult = self.args.scheduler_t_mult
        nr_restarts = self.args.nr_restarts
        weight_norm = self.args.weight_norm

        X_train = torch.tensor(X).float()
        y_train = torch.tensor(y)
        y_train = y_train.float() if self.nr_classes == 2 else y_train.long()
        X_train = X_train.to(self.dev)
        y_train = y_train.to(self.dev)

        # Create dataloader for training
        train_dataset = torch.utils.data.TensorDataset(
            X_train,
            y_train,
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # calculate the initial budget given the total number of iterations,
        # the number of restarts and the budget multiplier
        T_0: int = max(
            ((nr_epochs * len(train_loader)) * (scheduler_t_mult - 1)) // (scheduler_t_mult ** nr_restarts - 1), 1)

        # Train the hypernetwork
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0, scheduler_t_mult)

        # warmup the learning rate for 5 epochs
        def warmup(current_step: int):
            return float(current_step / (5 * len(train_loader)))

        scheduler1 = LambdaLR(optimizer, lr_lambda=warmup)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5 * len(train_loader)])

        if self.mode == 'classification':
            if self.nr_classes > 2:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.MSELoss()

        if not self.disable_wandb:
            wandb.watch(self.model, criterion, log='all', log_freq=10)

        ensemble_snapshot_intervals = [T_0, (scheduler_t_mult + 1) * T_0,
                                       (scheduler_t_mult ** 2 + scheduler_t_mult + 1) * T_0]
        iteration = 0
        loss_per_epoch = []

        train_auroc_per_epoch = []
        for epoch in range(1, nr_epochs + 1):

            loss_value = 0
            train_auroc = 0
            for batch_idx, batch in enumerate(train_loader):

                iteration += 1
                x, y = batch

                # pushing the model to evaluation mode in case fgsm is used for augmentation
                self.model.eval()

                info = augment_data(
                    x,
                    y,
                    self.numerical_features,
                    self.model,
                    criterion,
                    augmentation_prob=augmentation_probability,
                )

                self.model.train()
                optimizer.zero_grad()

                if len(info) == 4:
                    x, y_1, y_2, lam = info
                    if self.interpretable:
                        output, weights = self.model(x, return_weights=True)
                    else:
                        output = self.model(x)

                    if self.nr_classes == 2:
                        output = output.squeeze(1)

                    main_loss = lam * criterion(output, y_1) + (1 - lam) * criterion(output, y_2)
                else:
                    x, adversarial_x, y_1, y_2, lam = info
                    if self.interpretable:
                        output, weights = self.model(x, return_weights=True)
                        output_adv = self.model(adversarial_x)
                    else:
                        output = self.model(x)
                        output_adv = self.model(adversarial_x)

                    if self.nr_classes == 2:
                        output = output.squeeze(1)
                        output_adv = output_adv.squeeze(1)

                    main_loss = lam * criterion(output, y_1) + (1 - lam) * criterion(output_adv, y_2)

                if self.interpretable:
                    if self.nr_classes > 2:
                        weights = torch.squeeze(weights)
                    else:
                        weights = torch.squeeze(weights, dim=2)

                    weights = torch.abs(weights)
                    l1_loss = torch.mean(torch.flatten(weights))
                    loss = main_loss + (weight_norm * l1_loss)
                else:
                    loss = main_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                if self.nr_classes == 2:
                    batch_auroc = binary_auroc(output, y)
                else:
                    batch_auroc = multiclass_auroc(output, y, num_classes=self.nr_classes)

                loss_value += loss.item()
                train_auroc += batch_auroc

                if iteration in ensemble_snapshot_intervals:
                    self.ensemble_snapshots.append(deepcopy(self.model.state_dict()))

            loss_value /= len(train_loader)
            train_auroc /= len(train_loader)

            print(f'Epoch: {epoch}, Loss: {loss_value}, AUROC: {train_auroc}')
            loss_per_epoch.append(loss_value)
            train_auroc_per_epoch.append(train_auroc.detach().to('cpu').item())

            if not self.disable_wandb:
                wandb.log({"Train:loss": loss_value, "Train:auroc": train_auroc,
                           "Learning rate": optimizer.param_groups[0]['lr']})

        torch.save(self.model.state_dict(), os.path.join(self.output_directory, 'model.pt'))

        return self

    def predict(
        self,
        X_test: Union[List, np.ndarray, pd.DataFrame],
        y_test: Union[List, np.ndarray, pd.DataFrame] = None,
        return_weights: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predicts the output for the given input data.

        Args:
            X_test: list, np.ndarray or pd.DataFrame
                The input data for which the output should be predicted.
            y_test: list, np.ndarray or pd.DataFrame
                The ground truth labels for the given input data.
            return_weights: bool
                Whether to return the importance weights or not.

        Returns:
            predictions, weights: np.ndarray or Tuple[np.ndarray, np.ndarray]
                The predicted output for the given input data and the importance weights
                if requested.
        """
        # check if X_test is a DataFrame
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        elif isinstance(X_test, list):
            X_test = np.array(X_test)

        if y_test is not None:
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.to_numpy()
            elif isinstance(y_test, list):
                y_test = np.array(y_test)

        X_test = torch.tensor(X_test).float()
        X_test = X_test.to(self.dev)

        predictions = []
        weights = []
        for snapshot_idx, snapshot in enumerate(self.ensemble_snapshots):
            self.model.load_state_dict(snapshot)
            self.model.eval()

            if self.interpretable:
                output, model_weights = self.model(X_test, return_weights=True)
            else:
                output = self.model(X_test)

            output = output.squeeze(1)

            if self.mode == 'classification':
                if self.nr_classes > 2:
                    output = self.softmax_act_func(output)
                else:
                    output = self.sigmoid_act_func(output)

            predictions.append([output.detach().to('cpu').numpy()])
            if self.interpretable:
                weights.append([model_weights.detach().to('cpu').numpy()])

        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
        predictions = np.squeeze(predictions)

        if self.interpretable and return_weights:
            weights = np.array(weights)
            weights = np.squeeze(weights)
            if len(weights.shape) > 2:
                # take only the weights belonging to the last ensemble member
                weights = weights[-1, :, :]
                weights = np.squeeze(weights)

            # remove the bias weights
            weights = weights[:, :-1]

            if self.mode == 'classification':
                if self.nr_classes == 2:
                    act_predictions = (predictions > 0.5).astype(int)
                else:
                    act_predictions = np.argmax(predictions, axis=1)

                selected_weights = []
                correct_test_examples = []
                for test_example_idx in range(weights.shape[0]):
                    # if true test labels are provided, calculate the weight importances
                    # only for the correctly classified test examples.
                    if y_test is not None:
                        if y_test[test_example_idx] != act_predictions[test_example_idx]:
                            continue

                    # select the weights for the predicted class
                    if self.nr_classes > 2:
                        selected_weights.append(weights[test_example_idx, :, act_predictions[test_example_idx]])
                    else:
                        selected_weights.append(weights[test_example_idx, :])
                        correct_test_examples.append(test_example_idx)

                weights = np.array(selected_weights)
                correct_test_examples = np.array(correct_test_examples)

            test_examples = X_test.detach().to('cpu').numpy()
            correct_test_examples = test_examples[correct_test_examples]

            weights = weights * correct_test_examples
            weights = np.mean(np.abs(weights), axis=0)
            weights = weights / np.sum(weights)

        if self.interpretable:
            if return_weights:
                return predictions, weights
            else:
                return predictions
        else:
            return predictions
