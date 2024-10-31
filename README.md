# [NeurIPS 2024] Mesomorphic Interpretable Networks for Tabular Data

Even though neural networks have been long deployed in applications involving tabular data, still existing neural architectures are not explainable by design. In this paper, we propose a new class of interpretable neural networks for tabular data that are both deep and linear at the same time (i.e. mesomorphic). We optimize deep hypernetworks to generate explainable linear models on a per-instance basis. As a result, our models retain the accuracy of black-box deep networks while offering  free lunch explainability for tabular data by design. Through extensive experiments, we demonstrate that our explainable deep networks have comparable performance to state-of-the-art classifiers on tabular data and outperform current existing methods that are explainable by design.

Authors: Arlind Kadra, Sebastian Pineda Arango, and Josif Grabocka

## Setting up the virtual environment

```
# The following commands assume the user is in the cloned directory
conda create -n imn python=3.9
conda activate imn
cat requirements.txt | xargs -n 1 -L 1 pip install
```

## Running the code

The entry script to run IMN and TabResNet is `hpo_main_experiment.py`. 
The entry script to run the baseline methods (CatBoost, Random Forest, Logistic Regression, Decision Tree and TabNet) is `hpo_baseline_experiment.py`.

The main arguments for `hpo_main_experiment.py` are:

- `--nr_blocks`: Number of residual blocks in the hypernetwork.
- `--hidden_size`: The number of hidden units per-layer.
- `--augmentation_probability`: The probability with which data augmentation will be applied.
- `--scheduler_t_mult`: Number of restarts for the learning rate scheduler.
- `--seed`: The random seed to generate reproducible results.
- `--dataset_id`: The OpenML dataset id.
- `--test_split_size`: The fraction of total data that will correspond to the test set.
- `--nr_restarts`: Number of restarts for the learning rate scheduler.
- `--output_dir`: Directory where to store results.
- `--interpretable`: If interpretable results should be generated, basically if IMN should be used or the TabResNet architecture.
- `--mode`: Takes two arguments, `classification` and `regression`. 
- `--hpo_tuning`: Whether to enable hyperparameter optimization. 
- `--nr_trials`: The number of trials when performing hyperparameter optimization. 
- `--disable_wandb`: Whether to disable wandb logging. 



**A minimal example of running IMN**:

```
python hpo_main_experiment.py --hpo_tuning --n_trials 3 --disable_wandb --interpretable --dataset_id 1590

```


## Plots

The plots that are included in our paper were generated from the functions in the module `plots/comparison.py`.
The plots expect the following result folder structure:

```
├── results_folder
│   ├── method_name
│   │   ├── dataset_id
│   │   │   ├── seed
│   │   │   │   ├── output_info.json
```

## Citation

```
@inproceedings{
kadra2024interpretable,
title={Interpretable Mesomorphic Networks for Tabular Data},
author={Arlind Kadra and Sebastian Pineda Arango and Josif Grabocka},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=PmLty7tODm}
}
```