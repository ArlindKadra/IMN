from copy import deepcopy
import os
import time

import torch
import numpy as np
import pandas as pd
from models.hypernetwork import HyperNet
from models.dt_hypernetwork import DTHyperNet
from models.tabresnet import TabResNet
from models.dtree import DTree
from models.factorized_hypernet import FactorizedHyperNet
from dataset.neighbor_dataset import ContextDataset

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR, CosineAnnealingLR
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
        self.algorithm_backbone = {
            'tabresnet': TabResNet,
            'inn': HyperNet,
        }
        self.nr_classes = network_configuration['nr_classes'] if network_configuration['nr_classes'] != 1 else 2
        if model_name == 'inn':
            self.interpretable = True
        else:
            self.interpretable = False

        self.model_name = model_name
        self.network_configuration = network_configuration
        self.model = self.algorithm_backbone[model_name](**network_configuration)
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
        self.clip_value = 1.0
        self.mse_criterion = torch.nn.MSELoss()

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

        if self.mode == 'classification':
            y_train = torch.tensor(np.array(y)).float() if self.nr_classes == 2 else torch.tensor(
                np.array(y)).long()
        else:
            y_train = torch.tensor(np.array(y)).float()

        X_train = X_train.to(self.dev)
        y_train = y_train.to(self.dev)


        # Create dataloader for training
        train_dataset = torch.utils.data.TensorDataset(
            X_train,
            y_train,
        )
        #train_dataset = ContextDataset(X_train, y_train)
        #train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        T_0: int = max(
            ((nr_epochs * len(train_loader)) * (scheduler_t_mult - 1)) // (scheduler_t_mult ** nr_restarts - 1), 1)
        # Train the hypernetwork
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0, scheduler_t_mult)
        #scheduler2 = CosineAnnealingLR(optimizer, T_max=nr_epochs * len(train_loader))

        kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')
        def warmup(current_step: int):
            return float(current_step / (5 * len(train_loader)))

        scheduler1 = LambdaLR(optimizer, lr_lambda=warmup)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5 * len(train_loader)])
        #scheduler = CosineAnnealingLR(optimizer, T_max=nr_epochs * len(train_loader), eta_min=0.0001)
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
                #x, y, closest_x, closest_y = batch
                x, y = batch
                if self.mode != 'classification':
                    y = y.float()

                y = y.to(self.dev)
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
                optimizer.zero_grad(set_to_none=True)

                if len(info) == 4:
                    x, y_1, y_2, lam = info
                    if self.interpretable:
                        first_head_output, second_head_output, weights = self.model(x, return_weights=True)
                        #output, weights = self.model(x, return_weights=True)
                        #_, closest_weights = self.model(closest_x, return_weights=True)
                        #closest_output = torch.einsum("ij,ijk->ik", torch.cat((closest_x, torch.ones(x.shape[0], 1).to(x.device)), dim=1), weights)
                        #closest_output = self.model.calculate_predictions(closest_x, tree[0], tree[1], tree[2])
                        #_, _, tree = self.model(closest_x, return_weights=True, return_tree=True)
                        #closest_output = self.model.calculate_predictions(x, tree[0], tree[1], tree[2])
                        #feature_importances = main_tree[0]
                        #feature_importances = torch.cat(feature_importances, dim=0)
                        #feature_importances = torch.softmax(feature_importances, dim=1)
                        #entropy_loss = torch.mean(-feature_importances * torch.log(feature_importances))
                    else:
                        first_head_output = self.model(x)
                        #closest_output = self.model(closest_x)

                    if self.nr_classes == 2:
                        first_head_output = first_head_output.squeeze(1)
                        second_head_output = second_head_output.squeeze(1)
                        #closest_output = closest_output.squeeze(1)

                    main_loss = lam * criterion(first_head_output, y_1) + (1 - lam) * criterion(first_head_output, y_2)# + criterion(closest_output, closest_y)
                    if self.mode == 'classification':
                        if self.nr_classes > 2:
                            main_loss += kl_div_loss(torch.log(self.softmax_act_func(second_head_output)), self.softmax_act_func(first_head_output))
                        else:
                            main_loss += kl_div_loss(torch.log(self.sigmoid_act_func(second_head_output)), self.sigmoid_act_func(first_head_output))
                    else:
                        main_loss += self.mse_criterion(second_head_output, first_head_output)
                    #main_loss += self.mse_criterion(closest_output, output)
                    #main_loss += entropy_loss
                else:
                    x, adversarial_x, y_1, y_2, lam = info
                    if self.interpretable:
                        first_head_output, second_head_output, weights = self.model(x, return_weights=True)
                        #output, weights = self.model(x, return_weights=True)
                        #_, closest_weights = self.model(closest_x, return_weights=True)
                        #closest_output = torch.einsum("ij,ijk->ik", torch.cat((closest_x, torch.ones(x.shape[0], 1).to(x.device)), dim=1), weights)
                        #closest_output = self.model.calculate_predictions(closest_x, tree[0], tree[1], tree[2])
                        #_, _, tree = self.model(closest_x, return_weights=True, return_tree=True)
                        #closest_output = self.model.calculate_predictions(x, tree[0], tree[1], tree[2])
                        #feature_importances = main_tree[0]
                        #feature_importances = torch.cat(feature_importances, dim=0)
                        #feature_importances = torch.softmax(feature_importances, dim=1)
                        #entropy_loss = torch.mean(-feature_importances * torch.log(feature_importances))
                        output_adv = self.model(adversarial_x)
                    else:
                        output = self.model(x)
                        output_adv = self.model(adversarial_x)

                    if self.nr_classes == 2:
                        output = output.squeeze(1)
                        output_adv = output_adv.squeeze(1)
                        #closest_output = closest_output.squeeze(1)

                    main_loss = lam * criterion(output, y_1) + (1 - lam) * criterion(output_adv, y_2)# + criterion(closest_output, closest_y)
                    #main_loss += self.mse_criterion(closest_output, output)
                    #main_loss += entropy_loss

                if self.interpretable:


                    # take all values except the last one (bias)
                    if self.nr_classes > 2:
                        weights = torch.squeeze(weights)
                    else:
                        weights = torch.squeeze(weights, dim=2)

                    weights = torch.abs(weights)
                    #weights = torch.pow(weights, 2)
                    l1_loss = torch.mean(torch.flatten(weights))
                    if not torch.isnan(l1_loss):
                        main_loss += weight_norm * l1_loss

                # if main loss is nan
                if torch.isnan(main_loss):
                    print('nan loss')
                    self.model = self.algorithm_backbone[self.model_name](**self.network_configuration)
                    self.model = self.model.to(x.device)
                    continue

                main_loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

                optimizer.step()
                scheduler.step()

                if self.mode == 'classification':
                    # threshold the predictions if the model is binary
                    if self.nr_classes == 2:
                        predictions = (self.sigmoid_act_func(first_head_output) > 0.5).int()
                    else:
                        predictions = torch.argmax(first_head_output, dim=1)

                    # calculate balanced accuracy with pytorch
                    if self.nr_classes == 2:
                        batch_auroc = binary_auroc(first_head_output, y)
                    else:
                        batch_auroc = multiclass_auroc(first_head_output, y, num_classes=self.nr_classes)

                    train_auroc += batch_auroc
                else:
                    train_auroc = 0

                loss_value += main_loss.item()

                if iteration in ensemble_snapshot_intervals:
                    self.ensemble_snapshots.append(deepcopy(self.model.state_dict()))

            loss_value /= len(train_loader)
            train_auroc /= len(train_loader)

            print(f'Epoch: {epoch}, Loss: {loss_value}, AUROC: {train_auroc}')
            loss_per_epoch.append(loss_value)
            if self.mode == 'classification':
                train_auroc_per_epoch.append(train_auroc.detach().to('cpu').item())

            if not self.disable_wandb:
                wandb.log({"Train:loss": loss_value, "Train:auroc": train_auroc,
                           "Learning rate": optimizer.param_groups[0]['lr']})

            torch.save(self.model.state_dict(), os.path.join(self.output_directory, 'model.pt'))

        return self

    def predict(
            self,
            X_test,
            y_test=None,
            return_weights=False,
            return_tree=False,
            discretize=False,
    ):

        # check if X_test is a DataFrame
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()

        X_test = torch.tensor(X_test).float()

        X_test = X_test.to(self.dev)
        if y_test is not None:
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.to_numpy()
            y_test = torch.tensor(y_test).float() if self.nr_classes == 2 else torch.tensor(y_test).long()
        else:
            y_test = torch.zeros(X_test.size(0)).long()

        #y_test = y_test.to(self.dev)
        #test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        #test_dataset = ContextDataset(X_test, y_test)
        #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=X_test.size(0), shuffle=False)
        predictions = []
        weights = []

        for snapshot_idx, snapshot in enumerate(self.ensemble_snapshots):
            self.model.load_state_dict(snapshot)
            self.model.eval()
            if self.interpretable:
                if return_tree:
                    output, _, model_weights = self.model(X_test, return_weights=True)
                else:
                    output, _, model_weights = self.model(X_test, return_weights=True)
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
                #weights.append(model_weights.detach().to('cpu').numpy())
                weights.append(model_weights.detach().to('cpu').numpy())

        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
        # take only the last prediction
        #predictions = predictions[-1, :, :]
        predictions = np.squeeze(predictions)

        if self.interpretable and return_weights:
            weights = np.array(weights)
            weights = np.squeeze(weights)
            if len(weights.shape) > 2:
                weights = weights[-1, :, :]
                #weights = np.mean(weights, axis=0)
                weights = np.squeeze(weights)
            # take all values except the last one (bias)
            #weights = weights[:, :-1]
            #test_examples = X_test.detach().to('cpu').numpy()
            #weights = weights * test_examples

            if self.mode == 'classification':
                if self.nr_classes == 2:
                    act_predictions = (predictions > 0.5).astype(int)
                else:
                    act_predictions = np.argmax(predictions, axis=1)


                selected_weights = []
                #correct_test_examples = []
                for test_example_idx in range(weights.shape[0]):
                    # select the weights for the predicted class
                    if y_test[test_example_idx] == act_predictions[test_example_idx]:
                        if self.nr_classes > 2:
                            selected_weights.append(weights[test_example_idx, :, act_predictions[test_example_idx]])
                        else:
                            selected_weights.append(weights[test_example_idx, :])
                        #correct_test_examples.append(test_example_idx)
                weights = np.array(selected_weights)
                #correct_test_examples = np.array(correct_test_examples)

            """
            #weights_importances = generate_weight_importances_top_k(weights, 5)
            #
            # normalize the weights
            #weights_averages = weights_averages / np.sum(weights_averages)
            """
            #test_examples = X_test.detach().to('cpu').numpy()
            #correct_test_examples = test_examples[correct_test_examples]
            #weights = weights * correct_test_examples
            """
            #weights = np.mean(np.abs(weights), axis=0)
            #weights = weights / np.sum(weights)
            """
        #weights = np.mean(weights, axis=0)
        if self.interpretable:
            if return_weights:
                if return_tree:
                    return predictions, weights#, tree
                else:
                    return predictions, weights
            else:
                return predictions
        else:
            return predictions
