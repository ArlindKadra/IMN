import copy
import math

import torch
import numpy as np


import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
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
        "ytick.minor.visible" : True,
    },
    style="white"
)

import matplotlib.pyplot as plt
from models.hypernetwork import HyperNet
from models.dt_hypernetwork import DTHyperNet
from models.tabresnet import TabResNet
from torcheval.metrics.functional import binary_auroc, binary_accuracy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
first_feature = np.arange(0, 15, 0.5)
y = [math.sin(point) for point in first_feature]
from utils import generate_weight_importances_top_k

#plt.plot(first_feature, y)
#plt.xlabel('x1')
#plt.ylabel('x2')
#plt.show()

seed = 11
torch.manual_seed(seed)
np.random.seed(seed)
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

second_feature = np.arange(-2, 2.5, 0.5)

x_1_points = []
x_2_points = []
labels = []

negative_x1_points = []
negative_x2_points = []
posizive_x1_points = []
posizive_x2_points = []
for x_1 in first_feature:
    for x_2 in second_feature:

        if math.sin(x_1) < x_2:
            label = '1'
        elif math.sin(x_1) > x_2:
            label = '0'
        else:
            label = '-1'

        if label == '0' or label == '1':
            x_1_points.append(x_1)
            x_2_points.append(x_2)
            labels.append(float(label))

        if label == '1':
            posizive_x1_points.append(x_1)
            posizive_x2_points.append(x_2)
        elif label == '0':
            negative_x1_points.append(x_1)
            negative_x2_points.append(x_2)


x_1_points = np.array(x_1_points)
x_2_points = np.array(x_2_points)
x_1_points = np.expand_dims(x_1_points, 1)
x_2_points = np.expand_dims(x_2_points, 1)
labels = np.array(labels)

X_train = np.concatenate([x_1_points, x_2_points], axis=1)
"""
X_train, X_test, y_train, y_test = train_test_split(
    X_train,
    labels,
    test_size=0.2,
    random_state=seed,
    stratify=labels,
)
"""
fig, ax = plt.subplots(2, 2)
X_train, labels = make_moons(n_samples=1000, shuffle=True, random_state=seed, noise=0.1)
first_feature = X_train[:, 0]
second_feature = X_train[:, 1]
positive_examples = X_train[labels == 1]
negative_examples = X_train[labels == 0]
#ax[0, 0].scatter(positive_examples[:, 0], positive_examples[:, 1], label='Pos. Class', color='red', marker="^", s=2)
#ax[0, 0].scatter(negative_examples[:, 0], negative_examples[:, 1], label='Neg. Class', color='blue', marker="o", s=2)
#ax[0, 0].set_xlabel('$x_1$')
#ax[0, 0].set_ylabel('$x_2$')
#ax[0, 0].set_title('Original Data')

model = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
model.fit(X_train, labels)
y_pred = model.predict(X_train)
print(f'Random forest accuracy: {np.sum(y_pred == labels) / len(labels)}')
X_train = torch.tensor(X_train, dtype=torch.float32).to('cpu')
y_train = torch.tensor(labels, dtype=torch.float32).to('cpu')

batch_size = 32
train_dataset = torch.utils.data.TensorDataset(
    X_train,
    y_train,
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

hypernet = DTHyperNet(
    nr_features=X_train.size(1),
    nr_classes=1,
    nr_blocks=2,
    hidden_size=128,
).to('cpu')

criterion = torch.nn.BCEWithLogitsLoss()
second_criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(hypernet.parameters(), lr=0.01, weight_decay=1)
nr_epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nr_epochs)
hypernet.train()
specify_weight_norm = True
weight_norm = 0
for epoch in range(nr_epochs):
    epoch_loss = 0
    epoch_auroc = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        output, weights = hypernet(x, return_weights=True)
        output = output.squeeze()
        #weights = torch.squeeze(weights, dim=2)
        #weights = weights[:, :-1]
        l1_loss = torch.norm(weights, 1)
        main_loss = criterion(output, y)
        loss = main_loss #+ (weight_norm * l1_loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_auroc += binary_auroc(output, y)
    epoch_loss /= len(train_loader)
    epoch_auroc /= len(train_loader)
    scheduler.step()
    print(f'Epoch {epoch} loss: {epoch_loss}, accuracy: {epoch_auroc}')
"""
hypernet.eval()
y_test = y_test > 0.5
y_test = y_test.astype(np.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).to('cuda')
y_test = torch.tensor(y_test, dtype=torch.float32).to('cuda')
with torch.no_grad():
    output = hypernet(X_test)
    output = output.squeeze()
    loss = criterion(output, y_test)
    accuracy = binary_accuracy(output, y_test)
    print(f'Test loss: {loss}, accuracy: {accuracy}')

"""
#X_test = X_test.cpu().numpy()
#output = output.cpu().numpy()

hypernet.eval()
with torch.no_grad():
    output = hypernet(X_train)
    output = output.squeeze()

output = output.detach().cpu().numpy()
X_train = X_train.detach().cpu().numpy()
first_feature = X_train[:, 0]
second_feature = X_train[:, 1]
positive_examples_first_feature = []
positive_examples_second_feature = []
negative_examples_first_feature = []
negative_examples_second_feature = []

for instance_index, test_label in enumerate(output):
    test_value = sigmoid(test_label)
    if test_value > 0.5:
        positive_examples_first_feature.append(first_feature[instance_index])
        positive_examples_second_feature.append(second_feature[instance_index])
    else:
        negative_examples_first_feature.append(first_feature[instance_index])
        negative_examples_second_feature.append(second_feature[instance_index])

# create new figure
#plt.figure()
min_first_feature = np.min(first_feature)
max_first_feature = np.max(first_feature)
min_second_feature = np.min(second_feature)
max_second_feature = np.max(second_feature)

hyperplane_x_points = []
hyperplane_y_points = []
for point_first_feature in np.linspace(min_first_feature, max_first_feature, 100):
    output_list = []
    points = []
    for second_feature in np.linspace(min_second_feature, max_second_feature, 100):
        output = hypernet(torch.tensor([[point_first_feature, second_feature]], dtype=torch.float32).to('cpu'), discretize=True)
        output = output.squeeze()
        output = output.detach().cpu().numpy()
        output_list.append(output)
        points.append([point_first_feature, second_feature])
    output_list = np.abs(output_list)
    min_index = np.argmin(output_list)
    hyperplane_point = points[min_index]
    hyperplane_x_points.append(hyperplane_point[0])
    hyperplane_y_points.append(hyperplane_point[1])

ax[0, 0].scatter(positive_examples_first_feature, positive_examples_second_feature, color='red', marker="^", s=12)
ax[0, 0].scatter(negative_examples_first_feature, negative_examples_second_feature, color='blue', marker="o", s=12)
ax[0, 0].plot(hyperplane_x_points, hyperplane_y_points, label=r'$\{x\, | \, \hat{w}\left(x\right)^T\,x + \hat{w}_0\left(x\right) = 0\}$', color='green', linewidth=3.5)
ax[0, 0].set_xlabel('$x_1$')
ax[0, 0].set_ylabel('$x_2$')
ax[0, 0].set_title('Globally Accurate')
first_feature = np.arange(0, 15, 0.5)
y = [math.sin(point) for point in first_feature]
#plt.plot(first_feature, y, label='Global hyperplane', color='green')
#plt.savefig('hyperplane_model.pdf', bbox_inches='tight')

chosen_indices = np.random.choice(range(X_train.shape[0]), 2, replace=False)
chosen_examples = X_train[chosen_indices]
_, weights = hypernet(torch.tensor(chosen_examples).float().to('cpu'), return_weights=True)
weights = weights.detach().to('cpu').numpy()
#plt.figure()
ax[0, 1].scatter(positive_examples_first_feature, positive_examples_second_feature, color='red', marker="^", s=12)
ax[0, 1].scatter(negative_examples_first_feature, negative_examples_second_feature, color='blue', marker="o", s=12)
colors = ['black', 'orange']
markers = ['o', '^']
labels = ["$x^\mathrm{'}$", '$y=0$']
chosen_examples = [chosen_examples[1]]
for index, example in enumerate(chosen_examples):
    example_weights = weights[1]
    first_weight = example_weights[0]
    second_weight = example_weights[1]
    #bias_feature = example_weights[2]
    first_feature = example[0]
    second_feature = example[1]
    first_part_line = np.arange(-1.5, 2.5, 0.1)
    #second_part_line = [((first_weight * first_element) + bias_feature) / (-1 * second_weight) for first_element in first_part_line]
    second_part_line = [((first_weight * first_element)) / (-1 * second_weight) for first_element in first_part_line]
    refined_first_part_line = []
    refined_second_part_line = []
    for i in range(len(first_part_line)):
        if second_part_line[i] > -1.0 and second_part_line[i] < 1.6:
            refined_first_part_line.append(first_part_line[i])
            refined_second_part_line.append(second_part_line[i])
    ax[0, 1].plot(refined_first_part_line, refined_second_part_line, color=colors[index], label=r"$\{x \, | \, \hat{w}(x\mathrm{'})^T\,x + \hat{w}_0(x\mathrm{'}) = 0\}$", markersize=12, linewidth=3.5)
    ax[0, 1].scatter(first_feature, second_feature, color=colors[index], marker=markers[index], label=labels[index],
                     s=35)
    ax[0, 1].text(first_feature - 0.15, second_feature + 0.2, r"$x\mathrm{'}$", fontsize=27)
ax[0, 1].set_xlabel('$x_1$')
ax[0, 1].set_ylabel('$x_2$')
ax[0, 1].set_title('Locally Interpretable')
first_feature = np.arange(0, 15, 0.5)
y = [math.sin(point) for point in first_feature]


def legend_without_duplicate_labels(fig):
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    remove_duplicates = {}
    new_lines = []
    new_labels = []
    for i in range(len(lines)):
        if labels[i] not in remove_duplicates:
            remove_duplicates[labels[i]] = True
            new_lines.append(lines[i])
            new_labels.append(labels[i])
    fig.legend(new_lines, new_labels, loc='center left', bbox_to_anchor=(0.95, 0.7))
#fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.45), loc='lower center', ncol=3)

for i in range(1, 2):
    for j in range(2):
        ax[i, j].set_visible(False)
legend_without_duplicate_labels(fig)
fig.savefig('motivation.pdf', bbox_inches='tight')
"""
hypernet.eval()
X_train = torch.tensor(X_train).float().to('cuda')
with torch.no_grad():
    output, weights = hypernet(X_train, return_weights=True)
    weights = weights.squeeze_()
    weights = weights.detach().to('cpu').numpy()

X = weights[:, 0]
Y = weights[:, 1]
Z = weights[:, 2]
ax = plt.figure().add_subplot(projection='3d')

ax.scatter(X, Y, Z)
plt.savefig('3dweights.pdf', bbox_inches='tight')

nr_features = X_train.shape[1]
X_train, labels = make_moons(n_samples=1000, shuffle=True, random_state=seed, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(
    X_train,
    labels,
    test_size=0.2,
    random_state=seed,
    stratify=labels,
)
nr_random_features = [nr_features * i for i in range(1, 5)]
X_train_base = copy.deepcopy(X_train)
y_train = torch.tensor(y_train, dtype=torch.float32).to('cpu')
y_test = torch.tensor(y_test, dtype=torch.float32).to('cpu')

weight_norm = 0
first_feature_importances = []
second_feature_importances = []
legend_without_duplicate_labels(fig)
#plt.plot(first_feature, y, label='Global hyperplane', color='green')
fig.savefig('motivation.pdf', bbox_inches='tight')

for number_rfeatures in nr_random_features:

    # generate random noise
    if number_rfeatures > 0:
        noise = np.random.normal(0, 1, (X_train.shape[0], number_rfeatures))
    for model_name in ['inn']:

        if number_rfeatures > 0:
            # add noise to the data
            X_train_noise = np.concatenate((X_train, noise), axis=1)
            X_test_noise = np.concatenate((X_test, np.random.normal(0, 1, (X_test.shape[0], number_rfeatures))), axis=1)
        else:
            X_train_noise = copy.deepcopy(X_train)
            X_test_noise = copy.deepcopy(X_test)

        X_test_noise = torch.tensor(X_test_noise, dtype=torch.float32).to('cpu')
        X_train_new = torch.tensor(X_train_noise, dtype=torch.float32).to('cpu')
        train_dataset = torch.utils.data.TensorDataset(
            X_train_new,
            y_train,
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if model_name == 'tabresnet':
            model = TabResNet(
                nr_features=X_train_new.size(1),
                nr_classes=1,
                nr_blocks=2,
                hidden_size=128,
            ).to('cpu')
        else:
            model = HyperNet(
                nr_features=X_train_new.size(1),
                nr_classes=1,
                nr_blocks=2,
                hidden_size=128,
            ).to('cpu')

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        nr_epochs = 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nr_epochs)
        model.train()

        for epoch in range(nr_epochs):
            epoch_loss = 0
            epoch_auroc = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                if model_name == 'tabresnet':
                    output = model(x)
                else:
                    output, weights = model(x, return_weights=True)
                    weights = torch.squeeze(weights, dim=2)
                    weights = weights[:, :-1]
                    weights = torch.abs(weights)
                    l1_loss = torch.mean(weights)

                output = output.squeeze()
                main_loss = criterion(output, y)
                if model_name == 'inn':
                    loss = main_loss + (weight_norm * l1_loss)
                else:
                    loss = main_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_auroc += binary_auroc(output, y)
            epoch_loss /= len(train_loader)
            epoch_auroc /= len(train_loader)
            scheduler.step()
            print(f'Epoch {epoch} loss: {epoch_loss}, accuracy: {epoch_auroc}')

        model.eval()
        with torch.no_grad():
            output, weights = model(X_test_noise, return_weights=True)
            output = output.squeeze()
            test_auroc = binary_auroc(output, y_test)
            print(f'Test auroc: {test_auroc}')
        weights = weights.squeeze()
        weights = weights[:, :-1]
        weights = weights.detach().to('cpu').numpy()
        weights = weights * X_test_noise.detach().to('cpu').numpy()
        weights = np.abs(weights)
        weight_importances = np.mean(weights, axis=0)
        weight_importances = weight_importances / np.sum(weight_importances)
        first_feature_importances.append(weight_importances[0])
        second_feature_importances.append(weight_importances[1])
        print(f'First feature importance: {first_feature_importances}')
        print(f'Second feature importance: {second_feature_importances}')

#ax[0, 2].plot(nr_random_features, first_feature_importances, color='blue')
#ax[0, 2].plot(nr_random_features, second_feature_importances, color='red')
#ax[0, 2].set_title('Noise Resilience')
#ax[0, 2].set_xlabel('Nr. of Random Features')
#ax[0, 2].set_ylabel('$|\hat{w}|$')
#ax[0, 2].set_xticks(nr_random_features)
#ax[0, 2].text(4, 0.475, r"$|\hat{w}\left(x_2\right) x|$", fontsize=23, color='red')
#ax[0, 2].text(4, 0.44, r"$|\hat{w}\left(x_1\right) x|$", fontsize=23, color='blue')
#fig.subplots_adjust(wspace=0.55)
legend_without_duplicate_labels(fig)
#plt.plot(first_feature, y, label='Global hyperplane', color='green')
fig.savefig('motivation.pdf', bbox_inches='tight')
"""