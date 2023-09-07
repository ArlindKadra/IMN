import json
import os

import matplotlib.pyplot as plt
import openml

import seaborn as sns

import pandas as pd

import numpy as np

import scipy.stats as stats
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 31,
        'axes.titlesize': 31,
        'axes.labelsize': 31,
        'xtick.labelsize': 31,
        'ytick.labelsize': 31,
        'legend.fontsize': 31,
    },
    style="white"
)


def prepare_method_results(output_dir:str, method_name: str):

    result_dict = {
        'dataset_id': [],
        'train_auroc': [],
        'test_auroc': [],
    }
    method_output_dir = os.path.join(output_dir, method_name)
    for dataset_id in os.listdir(method_output_dir):
        dataset_dir = os.path.join(method_output_dir, dataset_id)
        seed_test_balanced_accuracy = []
        seed_train_balanced_accuracy = []
        for seed in os.listdir(dataset_dir):
            seed_dir = os.path.join(dataset_dir, seed)
            try:
                with open(os.path.join(seed_dir, 'output_info.json'), 'r') as f:
                    seed_result = json.load(f)
                    seed_test_balanced_accuracy.append(seed_result['test_auroc'])
                    seed_train_balanced_accuracy.append(seed_result['train_auroc'][-1] if method_name == 'inn' else seed_result['train_auroc'])
            except FileNotFoundError:
                print(f'No output_info.json found for {method_name} {dataset_id} {seed}')
        result_dict['dataset_id'].append(dataset_id)
        result_dict['train_auroc'].append(np.mean(seed_train_balanced_accuracy) if len(seed_train_balanced_accuracy) > 0 else np.NAN)
        result_dict['test_auroc'].append(np.mean(seed_test_balanced_accuracy) if len(seed_test_balanced_accuracy) > 0 else np.NAN)

    return pd.DataFrame.from_dict(result_dict)


def distribution_methods(output_dir: str, method_names: list):

    pretty_method_names = {
        'inn': 'INN',
        'inn_v2': 'INN 2',
        'random_forest': 'R. Forest',
        'catboost': 'CatBoost',
        'tabresnet': 'TabResNet',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'tabnet': 'TabNet',
    }
    method_results = []
    for method_name in method_names:
        method_results.append(prepare_method_results(output_dir, method_name))

    decision_tree_results = prepare_method_results(output_dir, 'decision_tree')
    pretty_names = [pretty_method_names[method_name] for method_name in method_names]

    # prepare distribution plot
    df_results = []

    for method_name, method_result in zip(method_names, method_results):
        # normalize method performances by decision tree performance for each dataset
        method_result['test_auroc'] = method_result['test_auroc'] / decision_tree_results['test_auroc']
        df_results.append(method_result.assign(method=method_name))

    df = pd.concat(df_results, axis=0)

    df['train_auroc'] = df['train_auroc'].fillna(0)
    df['test_auroc'] = df['test_auroc'].fillna(0)
    plt.boxplot([df[df['method'] == method_name]['test_auroc'] for method_name in method_names])
    plt.xticks(range(1, len(method_names) + 1), pretty_names)
    plt.ylabel('Gain')
    plt.savefig(os.path.join(output_dir, 'test_performance_comparison.pdf'), bbox_inches="tight")

def rank_methods(output_dir: str, method_names: list):

    inn_wins = 0
    catboost_wins = 0
    pretty_method_names = {
        'ordinal_inn': "Ordinal INN",
        'inn': 'INN',
        'inn_v2': 'INN 2',
        'random_forest': 'Random Forest',
        'catboost': 'CatBoost',
        'tabresnet': 'TabResNet',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'tabnet': 'TabNet',
    }
    pretty_names = [pretty_method_names[method_name] for method_name in method_names]

    method_results = []
    for method_name in method_names:
        method_results.append(prepare_method_results(output_dir, method_name))

    result_dfs = []
    for method_name, method_result in zip(method_names, method_results):
        result_dfs.append(method_result.assign(method=method_name))

    df = pd.concat(result_dfs, axis=0)
    method_ranks = dict()
    for method_name in method_names:
        method_ranks[method_name] = []

    catboost_performances = []
    inn_perfomances = []
    for dataset_id in df['dataset_id'].unique():
        method_dataset_performances = []
        try:
            considered_methods = []
            for method_name in method_names:
                # get test performance of method on dataset
                method_test_performance = df[(df['dataset_id'] == dataset_id) & (df['method'] == method_name)]['test_auroc'].values[0]
                method_dataset_performances.append(method_test_performance)
                if method_name == 'ordinal_inn':
                    considered_methods.append(method_test_performance)
                if method_name == 'random_forest':
                    considered_methods.append(method_test_performance)
                print(f'{method_name} {dataset_id}: {method_test_performance}')

            if len(considered_methods) == 2:
                inn_perfomances.append(considered_methods[0])
                catboost_performances.append(considered_methods[1])

            # generate ranks using scipy
            # convert lower to better
            method_dataset_performances = [-x for x in method_dataset_performances]
            ranks = stats.rankdata(method_dataset_performances, method='average')
        except IndexError:
            print(f'No test performance found for {dataset_id}')
            continue

        for rank_index, rank in enumerate(ranks):
            method_ranks[method_names[rank_index]].append(ranks[rank_index])


    # print mean rank for every method
    for method_name in method_names:
        print(f'{method_name}: {np.mean(method_ranks[method_name])}')

    # prepare distribution plot
    sns.violinplot(data=[method_ranks[method_name] for method_name in method_names])
    #plt.boxplot([method_ranks[method_name] for method_name in method_names])
    plt.xticks(range(0, len(method_names)), pretty_names)

    plt.ylabel('Rank')
    plt.savefig(os.path.join(output_dir, 'test_performance_rank_comparison.pdf'), bbox_inches="tight")
    # significance test
    print(stats.wilcoxon(catboost_performances, inn_perfomances))


def analyze_results(output_dir: str, method_names: list):

    inn = prepare_method_results(output_dir, 'inn')
    # pandas to csv
    inn.to_csv(os.path.join(output_dir, 'inn.csv'), index=False)

def prepare_cd_data(output_dir: str, method_names: list):

    pretty_method_names = {
        'inn': 'INN',
        'inn_v2': 'INN 2',
        'random_forest': 'Random Forest',
        'catboost': 'CatBoost',
        'tabresnet': 'TabResNet',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'tabnet': 'TabNet',
    }
    method_results = {}
    for method_name in method_names:
        # remove column from df
        method_result = prepare_method_results(output_dir, method_name).drop(columns=['train_auroc'])
        # convert from accuracy to error
        method_result['test_auroc'] = 1 - method_result['test_auroc']
        method_results[method_name] = method_result

    # prepare distribution plot
    df_results = []

    filtered_tasks = method_results['inn']['dataset_id']
    # get the common dataset ids between all methods
    #for method_name in method_names:
    #    filtered_tasks = set(filtered_tasks).intersection(set(method_results[method_name]['dataset_id']))

    for method_name in method_names:
        method_result = method_results[method_name]
        # only consider tasks that are in inn
        method_result = method_result[method_result['dataset_id'].isin(filtered_tasks)]
        # if missing tasks, add them with 0
        missing_tasks = set(filtered_tasks) - set(method_result['dataset_id'])
        if len(missing_tasks) > 0:
            missing_tasks = pd.DataFrame({'dataset_id': list(missing_tasks), 'test_auroc': [1] * len(missing_tasks)})
            method_result = pd.concat([method_result, missing_tasks], axis=0)
        df_results.append(method_result.assign(method=pretty_method_names[method_name]))

    df = pd.concat(df_results, axis=0)
    df['test_auroc'] = df['test_auroc'].fillna(1)
    df.to_csv(os.path.join(output_dir, 'cd_data.csv'), index=False)

def calculate_method_time(output_dir: str, method_name: str):

    result_dict = {
        'dataset_id': [],
        'time': [],

    }
    method_output_dir = os.path.join(output_dir, method_name)
    for dataset_id in os.listdir(method_output_dir):
        dataset_dir = os.path.join(method_output_dir, dataset_id)
        seed_times = []
        for seed in os.listdir(dataset_dir):
            seed_dir = os.path.join(dataset_dir, seed)
            try:
                with open(os.path.join(seed_dir, 'output_info.json'), 'r') as f:
                    seed_result = json.load(f)
                    seed_times.append(seed_result['time'])
            except FileNotFoundError:
                print(f'No output_info.json found for {method_name} {dataset_id} {seed}')
        result_dict['dataset_id'].append(dataset_id)
        result_dict['time'].append(np.mean(seed_times) if len(seed_times) > 0 else np.NAN)

    return pd.DataFrame.from_dict(result_dict)


def calculate_method_times(output_dir: str, method_names: list):

    method_dfs = dict()
    for method_name in method_names:
        method_df = calculate_method_time(output_dir, method_name)
        # take times as list
        method_dfs[method_name] = method_df

    dataset_ids = method_dfs['inn']['dataset_id']
    slow_down = []
    for dataset_id in dataset_ids:
        inn_performance = method_dfs['inn'][method_dfs['inn']['dataset_id'] == dataset_id]['time'].values[0]
        tabresnet = method_dfs['tabresnet'][method_dfs['tabresnet']['dataset_id'] == dataset_id]['time'].values[0]
        slow_down.append(inn_performance / tabresnet)
    print(f'Mean slow down: {np.mean(slow_down)}')
    print(f'Std slow down: {np.std(slow_down)}')

def prepare_result_table(output_dir: str, method_names: list, mode='test'):

    pretty_method_names = {
        'inn': 'INN',
        'inn_v2': 'INN 2',
        'random_forest': 'R. Forest',
        'catboost': 'CatBoost',
        'tabresnet': 'TabResNet',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'tabnet': 'TabNet',
    }
    method_results = []
    for method_name in method_names:
        method_results.append(prepare_method_results(output_dir, method_name))

    if mode == 'test':
        result_metric = 'test_auroc'
    else:
        result_metric = 'train_auroc'

    dataset_ids = method_results[-1]['dataset_id']

    method_info = {
        'dataset_id': [],
        'decision_tree': [],
        'logistic_regression': [],
        'random_forest': [],
        'tabnet': [],
        'tabresnet': [],
        'catboost': [],
        'inn': [],
    }
    for dataset_id in dataset_ids:
        method_info['dataset_id'].append(int(dataset_id))
        for method_name, method_result in zip(method_names, method_results):
            if dataset_id not in method_result['dataset_id'].values:
                method_info[method_name].append(-1)
            else:
                method_info[method_name].append(method_result[method_result['dataset_id'] == dataset_id][result_metric].values[0])


    df_results = pd.DataFrame.from_dict(method_info)
    # sort rows by dataset id
    df_results = df_results.sort_values(by='dataset_id')
    print(df_results.to_latex(index=False, float_format="%.3f"))

    dataset_info_dict = {
        'Dataset ID': [],
        'Dataset Name': [],
        'Number of Instances': [],
        'Number of Features': [],
        'Number of Classes': [],
        'Majority Class Percentage': [],
        'Minority Class Percentage': [],
    }
    for dataset_id in dataset_ids:
        dataset = openml.datasets.get_dataset(int(dataset_id), download_data=False)
        number_of_instances = dataset.qualities['NumberOfInstances']
        number_of_features = dataset.qualities['NumberOfFeatures']
        majority_class_percentage = dataset.qualities['MajorityClassPercentage']
        minority_class_percentage = dataset.qualities['MinorityClassPercentage']
        number_of_classes = dataset.qualities['NumberOfClasses']

        dataset_info_dict['Dataset ID'].append(int(dataset_id))
        dataset_info_dict['Dataset Name'].append(dataset.name)
        dataset_info_dict['Number of Instances'].append(int(number_of_instances))
        dataset_info_dict['Number of Features'].append(int(number_of_features))
        dataset_info_dict['Number of Classes'].append(int(number_of_classes))
        dataset_info_dict['Majority Class Percentage'].append(majority_class_percentage)
        dataset_info_dict['Minority Class Percentage'].append(minority_class_percentage)

    print(max(dataset_info_dict['Number of Instances']))
    print(min(dataset_info_dict['Number of Instances']))
    print(max(dataset_info_dict['Number of Features']))
    print(min(dataset_info_dict['Number of Features']))
    print(len(dataset_info_dict['Number of Classes']))
    df_dataset_info = pd.DataFrame.from_dict(dataset_info_dict)
    df_dataset_info = df_dataset_info.sort_values(by='Dataset ID')
    print(df_dataset_info.to_latex(index=False, float_format="%.3f"))

result_directory = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'inn_results',
    )
)

method_names = ['decision_tree', 'logistic_regression', 'random_forest', 'catboost', 'tabnet', 'tabresnet', 'inn']
#rank_methods(result_directory, method_names)
#prepare_cd_data(result_directory, method_names)
#analyze_results(result_directory, [])
#distribution_methods(result_directory, method_names)
#calculate_method_times(result_directory, method_names)

prepare_result_table(result_directory, method_names, mode='train')