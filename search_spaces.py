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