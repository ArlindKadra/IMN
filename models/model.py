from copy import deepcopy
import os
import time

import torch
import numpy as np
import pandas as pd
from models.hypernetwork import HyperNet
from models.tabresnet import TabResNet

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR
from utils import augment_data, generate_weight_importances_top_k
from torcheval.metrics.functional import binary_auroc, multiclass_auroc, binary_accuracy, multiclass_accuracy
import torch
import numpy as np
import wandb

class Classifier():

    def __init__(
            self,
            network_configuration,
            args,
            categorical_indicator,
            attribute_names,
            model_name='inn',
            device='cpu',
            mode='classification',
            output_directory='.',
            disable_wandb=True,
    ):
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
        self.mode = mode
        self.numerical_features = [i for i in range(len(categorical_indicator)) if not categorical_indicator[i]]
        self.attribute_names = attribute_names
        self.ensemble_snapshots = []
        self.sigmoid_act_func = torch.nn.Sigmoid()
        self.softmax_act_func = torch.nn.Softmax(dim=1)
        self.output_directory = output_directory

    def fit(self, X, y):

        # check if X_test is a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        nr_epochs = self.args.nr_epochs
        batch_size = self.args.batch_size
        learning_rate = self.args.learning_rate
        augmentation_probability = self.args.augmentation_probability
        weight_decay = self.args.weight_decay
        scheduler_t_mult = self.args.scheduler_t_mult
        nr_restarts = self.args.nr_restarts
        weight_norm = self.args.weight_norm

        X_train = torch.tensor(np.array(X)).float()
        y_train = torch.tensor(np.array(y)).float() if self.nr_classes == 2 else torch.tensor(
            np.array(y)).long()
        X_train = X_train.to(self.dev)
        y_train = y_train.to(self.dev)

        # Create dataloader for training
        train_dataset = torch.utils.data.TensorDataset(
            X_train,
            y_train,
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        T_0: int = max(
            ((nr_epochs * len(train_loader)) * (scheduler_t_mult - 1)) // (scheduler_t_mult ** nr_restarts - 1), 1)
        # Train the hypernetwork
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0, scheduler_t_mult)

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
                    # take all values except the last one (bias)
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

                # threshold the predictions if the model is binary
                if self.nr_classes == 2:
                    predictions = (self.sigmoid_act_func(output) > 0.5).int()
                else:
                    predictions = torch.argmax(output, dim=1)

                # calculate balanced accuracy with pytorch
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

    def predict(self, X_test, y_test=None):

        # check if X_test is a DataFrame
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
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
                weights.append([np.abs(model_weights.detach().to('cpu').numpy())])

        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
        predictions = np.squeeze(predictions)

        if self.interpretable:
            weights = np.array(weights)
            weights = np.squeeze(weights)
            if len(weights.shape) > 2:
                weights = weights[-1, :, :]
                weights = np.squeeze(weights)

            if self.mode == 'classification':
                if self.nr_classes == 2:
                    act_predictions = (predictions > 0.5).astype(int)
                else:
                    act_predictions = np.argmax(predictions, axis=1)

                weights = weights[:, :-1]
                selected_weights = []
                for test_example_idx in range(weights.shape[0]):
                    # select the weights for the predicted class
                    if y_test[test_example_idx] == act_predictions[test_example_idx]:
                        if self.nr_classes > 2:
                            selected_weights.append(weights[test_example_idx, :, predictions[test_example_idx]])
                        else:
                            selected_weights.append(weights[test_example_idx, :])

                weights = np.array(selected_weights)

            weights_importances = generate_weight_importances_top_k(weights, 5)

        if self.interpretable:
            return predictions, weights_importances
        else:
            return predictions
