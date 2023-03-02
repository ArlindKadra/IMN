import openml
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch

import numpy as np

from models.HyperNetwork import HyperNet

def main():

    # Get the data
    dataset = openml.datasets.get_dataset(1590)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )

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
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

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

    # Train a hypernetwork
    hypernet = HyperNet(nr_features, nr_classes)

    # Create dataloader for training
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train.values).float(),
        torch.tensor(y_train.values).long(),
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Train the hypernetwork
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            optimizer.zero_grad()
            output, weights = hypernet(x, return_weights=True)
            l1_loss = torch.norm(weights, 1)

            loss = criterion(output, y) + 0.001 * l1_loss
            loss.backward()
            optimizer.step()
            # print accuracy and loss per batch
            predictions = torch.argmax(output, dim=1)
            balanced_accuracy = balanced_accuracy_score(y, predictions)
            accuracy = accuracy_score(y, predictions)
            print("Epoch: %d, Batch: %d, Loss: %0.2f, Balanced accuracy: %0.2f, Accuracy: %0.2f" % (epoch, batch_idx, loss.item(), balanced_accuracy, accuracy))

    # calculate the accuracy of the hypernetwork
    predictions, weights = hypernet(torch.tensor(X_test.values).float(), return_weights=True)
    predictions = torch.argmax(predictions, dim=1)
    predictions = predictions.detach().numpy()
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print("Balanced accuracy: %0.2f" % balanced_accuracy)
    print("Accuracy: %0.2f" % accuracy)
    weights = weights.detach().numpy()
    selected_weights = []
    for test_example_idx in range(weights.shape[0]):
        selected_weights.append(weights[test_example_idx, :, predictions[test_example_idx]])

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
    main()
