import copy
import math

import torch
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib

matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
import seaborn as sns


sns.set_style('white')

sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 27,
        'axes.titlesize': 27,
        'axes.labelsize': 27,
        'xtick.labelsize': 27,
        'ytick.labelsize': 27,
        'legend.fontsize': 27,
        "xtick.bottom": True,
        "xtick.minor.visible": True,
        "ytick.left": True,
        "ytick.minor.visible": True,
    },
    style="white"
)

import matplotlib.pyplot as plt
from models.hypernetwork import HyperNet
from models.tabresnet import TabResNet
from torcheval.metrics.functional import binary_auroc, binary_accuracy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons


seed = 11
torch.manual_seed(seed)
np.random.seed(seed)


def sigmoid(input_array):
    thresholded_output = []
    for x in input_array:
        sigmoid_output = 1 / (1 + math.exp(-x))
        thresholded_output.append(sigmoid_output)

    return np.array(thresholded_output)


X_train, labels = make_moons(n_samples=1000, shuffle=True, random_state=seed, noise=0.1)

X_train = torch.tensor(X_train, dtype=torch.float32).to('cpu')
y_train = torch.tensor(labels, dtype=torch.float32).to('cpu')

batch_size = 64
train_dataset = torch.utils.data.TensorDataset(
    X_train,
    y_train,
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

hypernet = HyperNet(
    nr_features=X_train.size(1),
    nr_classes=1,
    nr_blocks=2,
    hidden_size=128,
    unit_type='basic',
).to('cpu')

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(hypernet.parameters(), lr=0.001)
nr_epochs = 1000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nr_epochs * len(train_loader))
hypernet.train()

weight_norm = 0
for epoch in range(nr_epochs):
    epoch_loss = 0
    epoch_auroc = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        output, weights = hypernet(x, return_weights=True, simple_weights=True)
        output = output.squeeze()

        l1_loss = torch.mean(torch.flatten(torch.abs(weights)))
        main_loss = criterion(output, y)
        loss = main_loss + (weight_norm * l1_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        epoch_auroc += binary_auroc(output, y)
    epoch_loss /= len(train_loader)
    epoch_auroc /= len(train_loader)
    print(f'Epoch {epoch} loss: {epoch_loss}, accuracy: {epoch_auroc}')

X_train = X_train.detach().cpu().numpy()


hypernet.eval()

k = 200
from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(X_train)

correct_examples = 0
total_examples = 0
for example_index, example in enumerate(X_train):
    # get the k closest point indices to the example
    closest_neighbors = neigh.kneighbors([example], return_distance=False, n_neighbors=1000)
    total_examples += k

    nr_positive_examples = int(k / 2)
    nr_negative_examples = k - nr_positive_examples

    balanced_neighbor_indices = []
    positive_indice = 0
    while nr_positive_examples > 0:
        if y_train[positive_indice] == 1:
            balanced_neighbor_indices.append(positive_indice)
            nr_positive_examples -= 1
        positive_indice += 1

    negative_indice = 0
    while nr_negative_examples > 0:
        if y_train[negative_indice] == 0:
            balanced_neighbor_indices.append(negative_indice)
            nr_negative_examples -= 1
        negative_indice += 1

    closest_examples = X_train[np.array(balanced_neighbor_indices)]

    example = torch.tensor([example], dtype=torch.float32).to('cpu')

    with torch.no_grad():
        _, weights = hypernet(example, return_weights=True, simple_weights=True)

    weights = weights.squeeze()
    weights = weights.detach().cpu().numpy()


    # repeat weights as many closest examples
    weights = np.stack([weights for _ in range(k)], axis=0)
    closest_examples = np.concatenate(
        (closest_examples,
            np.ones((closest_examples.shape[0], 1))
         ),
        axis=1,
    )
    closest_labels = y_train[balanced_neighbor_indices]
    # concatenate closest
    # elementwise multiplication
    output = np.multiply(closest_examples, weights)
    output = np.sum(output, axis=1)
    predictions = sigmoid(output)
    # threshold
    predictions = np.where(predictions > 0.5, 1, 0)
    for i in range(k):
        if predictions[i] == closest_labels[i]:
            correct_examples += 1

print(f'Accuracy: {correct_examples / total_examples}')