import json
import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import scipy.stats as stats
def prepare_method_results(output_dir:str, method_name: str):

    result_dict = {
        'dataset_id': [],
        'train_balanced_accuracy': [],
        'test_balanced_accuracy': [],
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
                    seed_test_balanced_accuracy.append(seed_result['test_balanced_accuracy'])
                    seed_train_balanced_accuracy.append(seed_result['train_balanced_accuracy'][-1] if method_name == 'inn' else seed_result['train_balanced_accuracy'])
            except FileNotFoundError:
                print(f'No output_info.json found for {method_name} {dataset_id} {seed}')
        result_dict['dataset_id'].append(dataset_id)
        result_dict['train_balanced_accuracy'].append(np.mean(seed_train_balanced_accuracy))
        result_dict['test_balanced_accuracy'].append(np.mean(seed_test_balanced_accuracy))

    return pd.DataFrame.from_dict(result_dict)


def distribution_methods(output_dir: str, method_names: list):

    pretty_method_names = {
        'inn': 'INN',
        'random_forest': 'Random Forest',
    }
    method_results = []
    for method_name in method_names:
        method_results.append(prepare_method_results(output_dir, method_name))

    pretty_names = [pretty_method_names[method_name] for method_name in method_names]

    # prepare distribution plot
    df = pd.DataFrame()
    for method_name, method_result in zip(method_names, method_results):
        df = df.append(method_result.assign(method=method_name))

    plt.boxplot([df[df['method'] == method_name]['test_balanced_accuracy'] for method_name in method_names])
    plt.xticks(range(1, len(method_names) + 1), pretty_names)
    plt.ylabel('Test balanced accuracy')
    plt.savefig(os.path.join(output_dir, 'test_performance_comparison.pdf'), bbox_inches="tight")

def rank_methods(output_dir: str, method_names: list):

    pretty_method_names = {
        'inn': 'INN',
        'random_forest': 'Random Forest',
    }
    pretty_names = [pretty_method_names[method_name] for method_name in method_names]

    method_results = []
    for method_name in method_names:
        method_results.append(prepare_method_results(output_dir, method_name))

    df = pd.DataFrame()
    for method_name, method_result in zip(method_names, method_results):
        df = df.append(method_result.assign(method=method_name))

    method_ranks = dict()
    for method_name in method_names:
        method_ranks[method_name] = []


    for dataset_id in df['dataset_id'].unique():
        method_dataset_performances = []
        for method_name in method_names:
            # get test performance of method on dataset
            method_test_performance = df[(df['dataset_id'] == dataset_id) & (df['method'] == method_name)]['test_balanced_accuracy'].values[0]
            method_dataset_performances.append(method_test_performance)
            print(f'{method_name} {dataset_id}: {method_test_performance}')
        # generate ranks using scipy
        ranks = stats.rankdata(method_dataset_performances, method='average')

        for rank_index, rank in enumerate(ranks):
            method_ranks[method_names[rank_index - 1]].append(ranks[rank_index])

    # print mean rank for every method
    for method_name in method_names:
        print(f'{method_name}: {np.mean(method_ranks[method_name])}')

    # prepare distribution plot
    plt.boxplot([method_ranks[method_name] for method_name in method_names])
    plt.xticks(range(1, len(method_names) + 1), pretty_names)
    plt.ylabel('Rank')
    plt.savefig(os.path.join(output_dir, 'test_performance_rank_comparison.pdf'), bbox_inches="tight")

def analyze_results(output_dir: str, method_names: list):

    inn = prepare_method_results(output_dir, 'inn')
    # pandas to csv
    inn.to_csv(os.path.join(output_dir, 'inn.csv'), index=False)

result_directory = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'inn_results',
    )
)

method_names = ['inn', 'random_forest']
rank_methods(result_directory, method_names)
analyze_results(result_directory, [])