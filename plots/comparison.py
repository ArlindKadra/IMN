import json
import os

import matplotlib.pyplot as plt
import openml

import seaborn as sns

import pandas as pd

import numpy as np

import scipy.stats as stats
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
sns.set(
    rc={
        #'figure.figsize': (11.7, 8.27),
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
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
                    seed_train_balanced_accuracy.append(seed_result['train_auroc'] if method_name == 'inn' else seed_result['train_auroc'])
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
                if method_name == 'inn':
                    considered_methods.append(method_test_performance)
                if method_name == 'tabresnet':
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
        'inn': 'IMN HPO',
        'inn_v2': 'INN 2',
        'random_forest': 'Random Forest HPO',
        'catboost': 'CatBoost HPO',
        'tabresnet': 'TabResNet HPO',
        'decision_tree': 'Decision Tree HPO',
        'logistic_regression': 'Logistic Regression HPO',
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

    #filtered_tasks = method_results['inn']['dataset_id']
    filtered_tasks = ['41164', '31', '3', '40984', '1067', '12', '41142', '54', '41146', '40981', '1489', '41143', '1464']
    # get the common dataset ids between all methods
    for method_name in method_names:
        filtered_tasks = set(filtered_tasks).intersection(set(method_results[method_name]['dataset_id']))

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
        'train_time': [],
        'inference_time': [],

    }
    method_output_dir = os.path.join(output_dir, method_name)
    for dataset_id in os.listdir(method_output_dir):
        dataset_dir = os.path.join(method_output_dir, dataset_id)
        seed_train_times = []
        seed_inference_times = []
        for seed in os.listdir(dataset_dir):
            seed_dir = os.path.join(dataset_dir, seed)
            try:
                with open(os.path.join(seed_dir, 'output_info.json'), 'r') as f:
                    seed_result = json.load(f)
                    seed_train_times.append(seed_result['train_time'])
                    seed_inference_times.append(seed_result['inference_time'])
            except FileNotFoundError:

                print(f'No output_info.json found for {method_name} {dataset_id} {seed}')
        if len(seed_train_times) == 0:
            continue
        result_dict['dataset_id'].append(dataset_id)
        result_dict['train_time'].append(np.median(seed_train_times) if len(seed_train_times) > 0 else np.NAN)
        result_dict['inference_time'].append(np.median(seed_inference_times) if len(seed_inference_times) > 0 else np.NAN)

    return pd.DataFrame.from_dict(result_dict)
from autorank import autorank, plot_stats, create_report, latex_table


def calculate_method_times(output_dir: str, method_names: list):

    method_dfs = dict()
    for method_name in method_names:
        method_df = calculate_method_time(output_dir, method_name)
        # take times as list
        method_dfs[method_name] = method_df

    # keep only dataset_ids that are common between all methods
    common_dataset_ids = set(method_dfs[method_names[0]]['dataset_id'])


    for method_name in method_names:
        # Mean training time
        print(f'{method_name} median training time: {np.median(method_dfs[method_name]["train_time"])}')
        # Mean inference time
        print(f'{method_name} median inference time: {np.median(method_dfs[method_name]["inference_time"])}')


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
        'hpo_run',
    )
)

method_names = ['decision_tree', 'logistic_regression', 'random_forest', 'catboost', 'tabnet', 'tabresnet', 'inn']
#rank_methods(result_directory, method_names)
#prepare_cd_data(result_directory, method_names)
#analyze_results(result_directory, [])
#distribution_methods(result_directory, method_names)
#calculate_method_times(result_directory, method_names)

def calculate_nr_trials(output_dir: str, method_name: str):

    result_dict = {
        'dataset_id': [],
        'nr_trials': [],

    }
    method_output_dir = os.path.join(output_dir, method_name)
    for dataset_id in os.listdir(method_output_dir):
        dataset_dir = os.path.join(method_output_dir, dataset_id)
        seed_dir = os.path.join(dataset_dir, '0')
        try:
            with open(os.path.join(seed_dir, 'trials.csv'), 'r') as f:
                trials_df = pd.read_csv(f)
                number_trials = len(trials_df)
        except FileNotFoundError:
            number_trials = 0
            print(f'No trials.csv found for {method_name} {dataset_id} {seed}')

        if number_trials == 0:
            continue

        result_dict['dataset_id'].append(dataset_id)
        result_dict['nr_trials'].append(number_trials)

    return pd.DataFrame.from_dict(result_dict)

def calculate_nr_trials_all_methods(output_dir: str, method_names: list):

        method_dfs = dict()
        for method_name in method_names:
            method_df = calculate_nr_trials(output_dir, method_name)
            # take times as list
            method_dfs[method_name] = method_df

        # keep only dataset_ids that are common between all methods
        common_dataset_ids = set(method_dfs[method_names[0]]['dataset_id'])
        for method_name in method_names:
            common_dataset_ids = common_dataset_ids.intersection(set(method_dfs[method_name]['dataset_id']))

        filtered_dfs = []
        # keep only the records that belong to the common dataset ids
        for method_name in method_names:
            filtered_dfs.append(method_dfs[method_name][method_dfs[method_name]['dataset_id'].isin(common_dataset_ids)])

        dataset_ids_high_nr_trials = []
        for dataset_id in common_dataset_ids:
            print(f'Dataset: {dataset_id}')
            number_trials = []
            for method_name, method_df in zip(method_names, filtered_dfs):
                number_trials.append(method_df[method_df["dataset_id"] == dataset_id]["nr_trials"].values[0])
                print(f'{method_name}: {method_df[method_df["dataset_id"] == dataset_id]["nr_trials"].values[0]}')
            if min(number_trials) > 20:
                dataset_ids_high_nr_trials.append(dataset_id)

        print(dataset_ids_high_nr_trials)

#calculate_method_times(result_directory, method_names)
import pandas as pd
from functools import reduce
def autorank_methods(output_dir: str, method_names: list):

    pretty_method_names = {
        'inn': 'IMN',
        'inn_v2': 'INN 2',
        'random_forest': 'Random Forest',
        'catboost': 'CatBoost',
        'tabresnet': 'TabResNet',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'tabnet': 'TabNet',
        'inn_dtree': 'INDTree',
        'nam': 'NAM',
        'inn_exp': 'INN',
    }
    method_results = {}
    for method_name in method_names:
        # remove column from df
        try:
            method_result = prepare_method_results(output_dir, method_name).drop(columns=['train_auroc'])
        except KeyError:
            method_result = prepare_method_results(output_dir, method_name)


        # convert from accuracy to error
        #method_result['test_auroc'] = 1 - method_result['test_auroc']
        method_results[method_name] = method_result

    # Step 1: Find the common dataset ids
    common_dataset_ids = set.intersection(*[set(df['dataset_id']) for df in method_results.values()])

    # Step 2: Filter each dataframe to only include the common dataset ids
    filtered_dfs = {method: df[df['dataset_id'].isin(common_dataset_ids)] for method, df in method_results.items()}

    # Step 3: Rename the columns appropriately for each method
    renamed_dfs = {
        method: df.rename(columns={'test_auroc': pretty_method_names[method]})
        for method, df in filtered_dfs.items()
    }

    # Step 4: Merge all the dataframes on the dataset id
    result = reduce(lambda left, right: pd.merge(left, right, on='dataset_id'), renamed_dfs.values())

    # drop the dataset id column
    result = result.drop(columns=['dataset_id'])
    result = autorank(result, alpha=0.05, verbose=False)
    plot_stats(result)
    print(result)
    plt.title('Average Rank')
    plt.savefig(os.path.join(output_dir, 'tuned_binary_black_box_cd.pdf'), bbox_inches="tight")

import openml

def prepare_result_table(output_dir: str, method_names: list, mode='test'):

    pretty_method_names = {
        'inn': 'IMN',
        'inn_v2': 'INN 2',
        'random_forest': 'R. Forest',
        'catboost': 'CatBoost',
        'tabresnet': 'TabResNet',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'tabnet': 'TabNet',
        'nam': 'NAM',
        'inn_dtree': 'INDTree',
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
        #'nam': [],
        'random_forest': [],
        'tabnet': [],
        'tabresnet': [],
        'catboost': [],
        'inn': [],
        #'inn_dtree': [],
    }
    for dataset_id in dataset_ids:
        method_info['dataset_id'].append(int(dataset_id))
        for method_name, method_result in zip(method_names, method_results):
            if dataset_id not in method_result['dataset_id'].values:
                method_info[method_name].append(-1)
            else:
                method_info[method_name].append(method_result[method_result['dataset_id'] == dataset_id][result_metric].values[0])


    df_results = pd.DataFrame.from_dict(method_info)
    # replace nan with -1
    df_results = df_results.fillna(-1)

    def highlight_max(df):
        styled_df = df.copy()
        for index, row in df.iterrows():
            max_val = row.max()
            styled_df.loc[index] = ['\\textbf{' + f"{val:.3f}" + '}' if val == max_val else f"{val:.3f}" for val in row]
        return styled_df

    df_results = df_results.sort_values(by='dataset_id')
    dataset_id_column = df_results['dataset_id']
    # exclude the dataset id column
    df_results = df_results.drop(columns=['dataset_id'])

    # Apply the function to the DataFrame
    df_results = highlight_max(df_results)
    # concatenate the dataset id column
    df_results = pd.concat([dataset_id_column, df_results], axis=1)
    # sort rows by dataset id

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

#autorank_methods(result_directory, method_names)
prepare_result_table(result_directory, method_names, 'train')

