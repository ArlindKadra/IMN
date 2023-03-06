from copy import deepcopy

import openml
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch

import numpy as np

from models.HyperNetwork import HyperNet
from utils import augment_data


def main(seed: int = 1):

    dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    cut_mix_prob = 0.5

    # Get the data
    dataset = openml.datasets.get_dataset(1590)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    numerical_features = [i for i in range(X.shape[1]) if not categorical_indicator[i]]
    # substitute missing values
    for i in range(X.shape[1]):
        if categorical_indicator[i] == False:
            X.iloc[:, i] = X.iloc[:, i].fillna(X.iloc[:, i].mean())
        else:
            categories = X.iloc[:, i].unique()
            # create a dictionary that maps each category to a number
            category_to_int = {category: i for i, category in enumerate(categories)}
            # replace the categories with numbers
            X.iloc[:, i] = X.iloc[:, i].replace(category_to_int)
            # add a new category -1 for missing values
            X.iloc[:, i] = X.iloc[:, i].cat.add_categories([-1])
            X.iloc[:, i] = X.iloc[:, i].fillna(-1)

    # standardize the data for numerical columns
    for i in range(X.shape[1]):
        if categorical_indicator[i] == False:
            X.iloc[:, i] = (X.iloc[:, i] - X.iloc[:, i].mean()) / X.iloc[:, i].std()

    # encode classes with numbers pandas
    y = y.astype('category')
    categories = y.unique()
    # create a dictionary that maps each category to a number
    category_to_int = {category: i for i, category in enumerate(categories)}
    # replace the categories with numbers
    y = y.replace(category_to_int)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    # calculate the balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    print("Balanced accuracy: %0.2f" % balanced_accuracy)
    print("Accuracy: %0.2f" % accuracy)

    # get random forest feature importances
    feature_importances = clf.feature_importances_
    print(feature_importances)
    # sort the feature importances in descending order
    sorted_idx = np.argsort(feature_importances)[::-1]
    # get the names of the top 10 features
    top_10_features = [attribute_names[i] for i in sorted_idx[:10]]
    print("Top 10 features: %s" % top_10_features)

    nr_features = X_train.shape[1]
    nr_classes = len(y_train.unique())

    network_configuration = {
        'nr_features': nr_features,
        'nr_classes': nr_classes,
        'nr_blocks': 2,
        'hidden_size': 128,
        'dropout_rate': 0.25,
    }

    # Train a hypernetwork
    hypernet = HyperNet(**network_configuration)
    hypernet = hypernet.to(dev)
    X_train = torch.tensor(X_train.values).float()
    y_train = torch.tensor(y_train.values).long()
    X_train = X_train.to(dev)
    y_train = y_train.to(dev)
    # Create dataloader for training
    train_dataset = torch.utils.data.TensorDataset(
        X_train,
        y_train,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    nr_epochs = 105
    # Train the hypernetwork
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 15, 2)
    criterion = torch.nn.CrossEntropyLoss()

    ensemble_snapshot_intervals = [15, 30, 60]
    ensemble_snapshots = []
    hypernet.train()
    for epoch in range(1, nr_epochs + 1):

        loss_value = 0
        accuracy_value = 0
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x, y_1, y_2, lam = augment_data(x, y, numerical_features, cut_mix_prob=0.2)
            optimizer.zero_grad()
            output, weights = hypernet(x, return_weights=True)
            l1_loss = torch.norm(weights, 1)
            loss = lam * criterion(output, y_1) + (1 - lam) * criterion(output, y_2) + 0.0001 * l1_loss
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(output, dim=1)

            # calculate accuracy with pytorch
            accuracy = torch.sum(predictions == y).item() / len(y)

            loss_value += loss.item()
            accuracy_value += accuracy

        loss_value /= len(train_loader)
        accuracy_value /= len(train_loader)
        print(f'Epoch: {epoch}, Loss: {loss_value}, Accuracy: {accuracy_value}')

        if epoch in ensemble_snapshot_intervals:
            ensemble_snapshots.append(deepcopy(hypernet.state_dict()))

        scheduler.step()

    snapshot_models = []
    for snapshot_idx, snapshot in enumerate(ensemble_snapshots):

        hypernet = HyperNet(**network_configuration)
        hypernet = hypernet.to(dev)
        hypernet.load_state_dict(snapshot)
        hypernet.eval()
        snapshot_models.append(hypernet)

    X_test = torch.tensor(X_test.values).float()
    X_test = X_test.to(dev)
    predictions = []
    weights = []
    for snapshot_idx, snapshot in enumerate(snapshot_models):
        with torch.no_grad():
            output, model_weights = snapshot(X_test, return_weights=True)
            predictions.append([output.detach().to('cpu').numpy()])
            weights.append([np.abs(model_weights.detach().to('cpu').numpy())])


    predictions = np.array(predictions)
    weights = np.array(weights)
    predictions = np.squeeze(predictions)
    weights = np.squeeze(weights)
    predictions = np.mean(predictions, axis=0)
    weights = np.mean(weights, axis=0)

    # from series to list
    y_test = y_test.tolist()
    predictions = np.argmax(predictions, axis=1)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print("Balanced accuracy: %0.2f" % balanced_accuracy)
    print("Accuracy: %0.2f" % accuracy)

    selected_weights = []
    for test_example_idx in range(weights.shape[0]):
        # select the weights for the predicted class and also take the absolute values
        if y_test[test_example_idx] == predictions[test_example_idx]:
            selected_weights.append(np.abs(weights[test_example_idx, :, predictions[test_example_idx]]))

    weights = np.array(selected_weights)
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


if __name__ == "__main__":
    seed = 11
    main(seed=seed)