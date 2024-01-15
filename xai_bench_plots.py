import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_style("white")
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 15,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
    },
    style="white"
)

base_path = os.path.expanduser(
    os.path.join(
            '~',
            'Desktop',
            'inn_results',
            'baselines',
    )
)

dataset_names = ['gaussianLinear', 'gaussianNonLinearAdditive', 'gaussianPiecewiseConstant']
pretty_dataset_names = ['G. Linear', 'G. Non-Linear', 'G. Piecewise Constant']
correlation_values = [0.0, 0.25, 0.50, 0.75, 0.99]
pretty_method_names = ['INN', 'K. SHAP', 'SHAP', 'LIME', 'Maple', 'L2X', 'BreakD.', 'TabNet', 'Random']
fig, ax = plt.subplots(4, 4)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.7)
metric_names = ['roar_faithfulness', 'roar_monotonicity', 'faithfulness', 'infidelity']
pretty_metric_names = ['Faithfulness (R)', 'Monotonicity (R)', 'Faithfulness', 'Infidelity']
for dataset_index, dataset_name in enumerate(dataset_names):
    for metric_index, metric_name in enumerate(metric_names):

        method_info = {
            'inn': [],
            'brutekernelshap': [],
            'shap': [],
            'lime': [],
            'maple': [],
            'l2x': [],
            'breakdown': [],
            'tabnet': [],
            'random': [],
        }
        for correlation_value in correlation_values:
            baselines_file_path = os.path.join(base_path, 'csv', f'{dataset_name}_{correlation_value}.csv')
            l2x_file_path = os.path.join(base_path, 'l2x', f'{dataset_name}_{correlation_value}.csv')
            inn_file_path = os.path.join(base_path, 'inn', f'{dataset_name}_{correlation_value}.csv')
            tabnet_file_path = os.path.join(base_path, 'tabnet', 'csv', f'{dataset_name}_{correlation_value}.csv')
            # pandas take second row as header

            baseline_pd = pd.read_csv(baselines_file_path, header=1)
            l2x_pd = pd.read_csv(l2x_file_path, header=1)
            inn_pd = pd.read_csv(inn_file_path, header=1)
            tabnet_pd = pd.read_csv(tabnet_file_path, header=1)
            result_file = pd.concat([baseline_pd, l2x_pd, inn_pd, tabnet_pd], axis=0)

            method_names = result_file.iloc[:, 0]
            method_names = method_names.tolist()
            roar_faithfulness = result_file[metric_name]
            for method_name, roar_value in zip(method_names, roar_faithfulness):
                method_info[method_name].append(roar_value)

        for method_index, method_name in enumerate(method_info):
            ax[dataset_index, metric_index].plot(correlation_values, method_info[method_name], label=pretty_method_names[method_index], linewidth=3)
        ax[dataset_index, metric_index].set_xlabel('Correlation')
        if dataset_index == 1:
            ax[dataset_index, metric_index].set_title(pretty_metric_names[metric_index])
        ax[dataset_index, metric_index].set_xticks([0, 0.5, 1])
        if metric_name == 'infidelity':
            ax[dataset_index, metric_index].set_ylim([0, 0.02])
        ax[dataset_index, metric_index].set_xticklabels([0, 0.5, 1])

    ax[dataset_index, 0].set_ylabel(pretty_dataset_names[dataset_index])
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
    fig.legend(new_lines, new_labels, bbox_to_anchor=(0.5, 0.1), loc='lower center', ncol=5)
    #fig.legend(new_lines, new_labels, bbox_to_anchor=(0.5, 0.57), loc='lower center', ncol=5)
"""
for i in range(1, 4):
    for j in range(4):
        ax[i, j].set_visible(False)
"""
for i in [0, 3]:
    for j in range(4):
        ax[i, j].set_visible(False)

legend_without_duplicate_labels(fig)
fig.savefig('xaibench.pdf', bbox_inches='tight')