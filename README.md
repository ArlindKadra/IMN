# Breaking the Paradox of Explainable Deep Learning

Deep Learning has achieved tremendous results by pushing the frontier of automation in diverse domains. Unfortunately, current neural network architectures are not explainable by design. In this paper, we propose a novel method that trains deep hypernetworks to generate explainable linear models. Our models retain the accuracy of black-box deep networks while offering free lunch explainability by design. Specifically, our explainable approach requires the same runtime and memory resources as black-box deep models, ensuring practical feasibility. Through extensive experiments, we demonstrate that our explainable deep networks are as accurate as state-of-the-art classifiers on tabular data. On the other hand, we showcase the interpretability of our method on a recent benchmark for empirically comparing prediction explainers. The experimental results reveal that our models are not only as accurate as their black-box deep-learning counterparts but also as interpretable as state-of-the-art explanation techniques.

NeurIPS Submission 697

## Setting up the virtual environment

```
# The following commands assume the user is in the cloned directory
conda create -n inn python=3.9
conda activate inn
cat requirements.txt | xargs -n 1 -L 1 pip install
```

## Running the code

The entry script to run INN and TabResNet is `main_experiment.py`. 
To entry script to run the baseline methods (CatBoost, Random Forest, Logistic Regression, Decision Tree and TabNet) is `baseline_experiment.py`.

The main arguments for `main_experiment.py` are:

- `--nr_blocks`: Number of residual blocks in the hypernetwork.
- `--hidden_size`: The number of hidden units per-layer.
- `--nr_epochs`: The number of epochs to train the hypernetwork.
- `--batch_size`: The number of examples in a batch.
- `--learning_rate`: The learning rate used during optimization.
- `--augmentation_probability`: The probability with which data augmentation will be applied.
- `--weight_decay`: The weight decay value.
- `--weight_norm`: The L1 coefficient that controls the sparsity induced in the final importances per-feature.
- `--scheduler_t_mult`: Number of restarts for the learning rate scheduler.
- `--seed`: The random seed to generate reproducible results.
- `--dataset_id`: The OpenML dataset id.
- `--test_split_size`: The fraction of total data that will correspond to the test set.
- `--nr_restarts`: Number of restarts for the learning rate scheduler.
- `--output_dir`: Directory where to store results.
- `--interpretable`: If interpretable results should be generated, basically if INN should be used or the TabResNet architecture.
- `--mode`: Takes two arguments, `classification` and `regression`. 



**A minimal example of running INN**:

```
python main_experiment.py --output_dir "." --dataset_id 1590 --nr_restarts 3 --weight_norm 0.1 --weight_decay 0.01 --seed 0 --interpretable

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
