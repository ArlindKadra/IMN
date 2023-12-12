import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons
import torch
import wandb

from models.model import Classifier

from matplotlib import pyplot as plt
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

def main(args: argparse.Namespace):

    dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    """"""
    """
    x_1_split = 0.4
    x_2_split = 0.6
    data = np.random.randn(10000, 5)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    labels = []
    for data_point in data:
        if data_point[0] <= x_1_split:
            if data_point[1] <= x_2_split:
                labels.append(1)
            else:
                labels.append(0)
        else:
            if data_point[1] <= x_2_split:
                labels.append(0)
            else:
                labels.append(1)

    labels = np.array(labels)
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    weight_importances = None
    test_split_size = args.test_split_size
    seed = args.seed
    X_train, labels = make_moons(n_samples=10000, shuffle=True, random_state=seed, noise=0.1)
    X_train = MinMaxScaler().fit_transform(X_train)
    X_train, X_test, y_train, y_test = train_test_split(X_train, labels, test_size=test_split_size, random_state=seed)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    attribute_names = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5']
    categorical_indicator = ['False', 'False', 'False', 'False', 'False']

    dataset_name = 'synthetic'
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
        '69',
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
        test_predictions, weight_importances, _ = model.predict(X_test, y_test, return_weights=True, return_tree=True)
        train_predictions, _, _ = model.predict(X_train, y_train, return_weights=True, return_tree=True)
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
            # normalize the predictions
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

    # create fine grid of points for both features between 0 and 1
    first_feature = np.arange(0, 1, 0.01)
    second_feature = np.arange(0, 1, 0.01)
    first_feature, second_feature = np.meshgrid(first_feature, second_feature)
    first_feature = first_feature.reshape(-1)
    second_feature = second_feature.reshape(-1)
    # create the input data
    input_data = np.stack((first_feature, second_feature), axis=1)
    # get the predictions
    predictions, weight_importances, trees = model.predict(
        torch.tensor(input_data).float().to('cpu'),
        return_weights=True,
        return_tree=True,
        discretize=True,
    )
    # threshold predictions
    if nr_classes == 2:
        # apply non torch sigmoid
        #predictions = 1 / (1 + np.exp(-predictions))
        predictions = (predictions > 0.5).astype(int)
    else:
        predictions = np.argmax(predictions, axis=1)

    # plot the predictions
    # add alpha value to colors
    plt.scatter(first_feature, second_feature, c=predictions, cmap='coolwarm', alpha=0.2, zorder=0)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', zorder=1)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    """
    feature_indices = tree[0]

    feature_importances = []
    for i in range(0, len(tree[0])):
        feature_importances.append(np.squeeze(tree[0][i].detach().cpu().numpy()))

    feature_thresholds = []
    for i in range(0, len(tree[1])):
        feature_thresholds.append(np.squeeze(tree[1][i].detach().cpu().numpy()))


    leaf_classes = []

    for i in range(0, len(tree[2])):
        class_value = np.squeeze(tree[2][i].detach().cpu().numpy())
        if class_value < 0.5:
            leaf_classes.append(0)
        else:
            leaf_classes.append(1)

    max_index = np.argmax(feature_importances[0])
    first_threshold = feature_thresholds[0][max_index]
    if max_index == 0:
        # plot vertical line on the first feature
        plt.axvline(x=first_threshold, color='red', linestyle='--', label='Vertical Line at x=3')
    else:
        # plot horizontal line on the second feature
        plt.axhline(y=first_threshold, color='red', linestyle='--', label='Horizontal Line at y=3')

    # plot the second decision boundary
    max_second_index = np.argmax(feature_importances[1])
    second_threshold = feature_thresholds[1][max_second_index]

    if max_second_index == 0:
        # plot vertical line on the first feature
        plt.axvline(x=second_threshold, color='red', linestyle='--', label='Vertical Line at x=3')
    else:
        # plot horizontal line on the second feature
        plt.axhline(y=second_threshold, color='red', linestyle='--', label='Horizontal Line at y=3')

    # plot the third decision boundary
    max_third_index = np.argmax(feature_importances[2])
    third_threshold = feature_thresholds[2][max_third_index]

    if max_third_index == 0:
        # plot vertical line on the first feature
        plt.axvline(x=third_threshold, color='red', linestyle='--', label='Vertical Line at x=3')
    else:
        # plot horizontal line on the second feature
        plt.axhline(y=third_threshold, color='red', linestyle='--', label='Horizontal Line at y=3')

    """
    # first_part_line = np.arange(-1.5, 2.5, 0.1)
    # second_part_line = [((first_weight * first_element) + bias_feature) / (-1 * second_weight) for first_element in first_part_line]
    # second_part_line = [((first_weight * first_element)) / (-1 * second_weight) for first_element in first_part_line]
    # refined_first_part_line = []
    # refined_second_part_line = []
    # for i in range(len(first_part_line)):
    #    if second_part_line[i] > -1.0 and second_part_line[i] < 1.6:
    #        refined_first_part_line.append(first_part_line[i])
    #        refined_second_part_line.append(second_part_line[i])
    # ax[0, 1].plot(refined_first_part_line, refined_second_part_line, color=colors[index], label=r"$\{x \, | \, \hat{w}(x\mathrm{'})^T\,x + \hat{w}_0(x\mathrm{'}) = 0\}$", markersize=12, linewidth=3.5)
   # ax[0, 1].scatter(first_feature, second_feature, color=colors[index], marker=markers[index], label=labels[index],
   #                  s=35)
    """
    root_node_feature = feature_indices[0]
    root_node_threshold = feature_thresholds[0]

    first_feature_ranges = np.linspace(-2, 2, 100)
    second_feature_ranges = np.linspace(-1, 1, 100)
    r1_x = np.linspace(-2, root_node_threshold, 100)
    r1_y_min = np.repeat(feature_thresholds[2], 100)
    r1_y_max = np.repeat(1, 100)

    r2_x = np.linspace(root_node_threshold, 2, 100)
    r2_y_min = np.repeat(feature_thresholds[1], 100)
    r2_y_max = np.repeat(1, 100)

    r3_x = np.linspace(-2, root_node_threshold, 100)
    r3_y_min = np.repeat(-1, 100)
    r3_y_max = np.repeat(feature_thresholds[2], 100)

    r4_x = np.linspace(root_node_threshold, 2, 100)
    r4_y_min = np.repeat(-1, 100)
    r4_y_max = np.repeat(feature_thresholds[1], 100)

    color_map = {1: 'red', 0: 'blue'}

    ax[0, 1].fill_between(r1_x, r1_y_min, r1_y_max, color=color_map[leaf_classes[2]], alpha=0.5)
    ax[0, 1].fill_between(r2_x, r2_y_min, r2_y_max, color=color_map[leaf_classes[0]], alpha=0.5)
    ax[0, 1].fill_between(r3_x, r3_y_min, r3_y_max, color=color_map[leaf_classes[3]], alpha=0.5)
    ax[0, 1].fill_between(r4_x, r4_y_min, r4_y_max, color=color_map[leaf_classes[1]], alpha=0.5)
    # set limits
    ax[0, 1].set_xlim((-2, 2))
    ax[0, 1].set_ylim((-1, 1))

    # ax[0, 1].text(first_feature - 0.15, second_feature + 0.2, r"$x\mathrm{'}$", fontsize=27)
    ax[0, 1].set_xlabel('$x_1$')
    ax[0, 1].set_ylabel('$x_2$')
    ax[0, 1].set_title('Locally Interpretable')
    first_feature = np.arange(0, 15, 0.5)
    y = [math.sin(point) for point in first_feature]
    """
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
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'decision_boundary_hdtree.pdf'))
    plt.close()

    chosen_example_index = int(0.25 * X_train.shape[0])
    chosen_example = X_train[chosen_example_index]
    # X_train to tensor
    X_train = torch.tensor(X_train).float()
    _, _, trees = model.predict(X_train, y_train, return_weights=True, return_tree=True)

    feature_importances = trees[0]
    feature_splits = trees[1]
    leaf_node_classes = trees[2]

    new_feature_importances = []
    new_feature_splits = []
    new_leaf_node_classes = []

    for i in range(0, len(feature_importances)):
        new_feature_importances.append(feature_importances[i][chosen_example_index, :].repeat(input_data.shape[0], 1))

    for i in range(0, len(feature_splits)):
        new_feature_splits.append(feature_splits[i][chosen_example_index, :].repeat(input_data.shape[0], 1))

    for i in range(0, len(leaf_node_classes)):
        new_leaf_node_classes.append(leaf_node_classes[i][chosen_example_index, :].repeat(input_data.shape[0], 1))


    input_data = torch.tensor(input_data).float().to('cuda:0')

    outputs = model.model.calculate_predictions(input_data, new_feature_importances, new_feature_splits, new_leaf_node_classes, discretize=True)
    # apply sigmoid
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu().numpy()
    # threshold the predictions
    if nr_classes == 2:
        outputs = (outputs > 0.5).astype(int)
    else:
        outputs = np.argmax(outputs, axis=1)

    plt.scatter((chosen_example[0]), (chosen_example[1]), c='black', s=100, zorder=2)
    plt.scatter(first_feature, second_feature, c=outputs, cmap='coolwarm', alpha=0.2, zorder=0)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', zorder=1)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'decision_boundary_hdtree_chosen_example_updated.pdf'))

    if interpretable:
        if weight_importances is not None:
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
        default=512,
        help="Batch size",
    )
    parser.add_argument(
        '--tree_depth',
        type=int,
        default=2,
        help='The depth of the tree.',
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
        default=0.0001,
        help="Weight decay",
    )
    parser.add_argument(
        "--weight_norm",
        type=float,
        default=0,
        help="Weight norm",
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
        default=1489,
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
    parser.add_argument(
        '--mode',
        type=str,
        default='classification',
        help='If we are doing classification or regression.',
    )

    args = parser.parse_args()

    main(args)

