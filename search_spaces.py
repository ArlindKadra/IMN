def hpo_space_imn(trial: optuna.trial.Trial) -> Dict:

    params = {
        'nr_epochs': trial.suggest_int('nr_epochs', 10, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'weight_norm': trial.suggest_float('weight_norm', 1e-5, 1e-1, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.5),
    }

    return params

def hpo_space_tabresnet(trial: optuna.trial.Trial) -> Dict:

    params = {
        'nr_epochs': trial.suggest_int('nr_epochs', 10, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.5),
    }

    return params

def hpo_space_logistic(trial: optuna.trial.Trial) -> Dict:

    params = {
        'C': trial.suggest_float('C', 1e-5, 5),
        'penalty': trial.suggest_categorical('penalty', ['l2', 'none']),
        'max_iter': trial.suggest_int('max_iter', 50, 500),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }

    return params

def hpo_space_dtree(trial: optuna.trial.Trial) -> Dict:

    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 21),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 11),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 3, 26),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
    }

    return params

def hpo_space_catboost(trial: optuna.trial.Trial) -> Dict:

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
        'random_strength': trial.suggest_int('random_strength', 1, 20),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-6, 1, log=True),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 20),
        'iterations': trial.suggest_int('iterations', 100, 4000)

    }

    return params

def hpo_space_random_forest(trial: optuna.trial.Trial) -> Dict:

    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 21),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 11),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 3, 26),
        'n_estimators': trial.suggest_int('n_estimators', 100, 4000),
    }

    return params

def hpo_space_tabnet(trial: optuna.trial.Trial) -> Dict:

    params = {
        'n_a': trial.suggest_categorical('n_a', [8, 16, 24, 32, 64, 128]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.02, 0.025]),
        'gamma': trial.suggest_categorical('gamma', [1.0, 1.2, 1.5, 2.0]),
        'n_steps': trial.suggest_categorical('n_steps', [3, 4, 5, 6, 7, 8, 9, 10]),
        'lambda_sparse': trial.suggest_categorical('lambda_sparse', [0, 0.000001, 0.0001, 0.001, 0.01, 0.1]),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]),
        'virtual_batch_size': trial.suggest_categorical('virtual_batch_size', [256, 512, 1024, 2048, 4096]),
        'decay_rate': trial.suggest_categorical('decay_rate', [0.4, 0.8, 0.9, 0.95]),
        'decay_iterations': trial.suggest_categorical('decay_iterations', [500, 2000, 8000, 10000, 20000]),
        'momentum': trial.suggest_categorical('momentum', [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]),
        'epochs': trial.suggest_int('epochs', 10, 500),
    }

    return params
