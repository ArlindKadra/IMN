import json
import os
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn

from nam.config import defaults
from nam.models import NAM, get_num_units
from nam.trainer import LitNAM
from nam.utils import *
from utils import get_dataset


def main(args: argparse.Namespace):

    dataset_id = args.dataset_id
    test_split_size = args.test_split_size
    seed = args.seed
    nr_epochs = args.nr_epochs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    info = get_dataset(
        dataset_id,
        test_split_size=test_split_size,
        seed=seed,
        encoding_type=args.encoding_type,
    )

    X_train = info['X_train']
    X_test = info['X_test']

    X_train = X_train.to_numpy()
    X_train = X_train.astype(np.float32)
    X_test = X_test.to_numpy()
    X_test = X_test.astype(np.float32)

    y_train = info['y_train']
    y_test = info['y_test']

    attribute_names = info['attribute_names']
    categorical_indicator = info['categorical_indicator']
    # the reference to info is not needed anymore
    del info

    X_train = torch.tensor(np.array(X_train)).float()
    X_test = torch.tensor(np.array(X_test)).float()
    nr_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    nr_classes = len(unique_classes)
    y_train = torch.tensor(np.array(y_train)).float() if nr_classes == 2 else torch.tensor(np.array(y_train)).long()
    y_test = torch.tensor(np.array(y_test)).float() if nr_classes == 2 else torch.tensor(np.array(y_test)).long()
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)
    X_train = X_train.to(dev)
    y_train = y_train.to(dev)
    X_test = X_test.to(dev)
    y_test = y_test.to(dev)

    # Create dataloader for training
    train_dataset = torch.utils.data.TensorDataset(
        X_train,
        y_train,
    )
    test_dataset = torch.utils.data.TensorDataset(
        X_test,
        y_test,
    )
    config = defaults()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    output_directory = os.path.join(
        args.output_dir,
        'nam',
        f'{args.dataset_id}',
        f'{args.seed}',

    )
    os.makedirs(output_directory, exist_ok=True)

    start_time = time.time()

    model = NAM(
      config=config,
      name="NAM_GALLUP",
      num_inputs=nr_features,
      num_units=get_num_units(config, X_train),
    )
    model.to(dev)

    config.num_epochs = nr_epochs
    config.seed = seed

    checkpoint_callback = ModelCheckpoint(
        filename=f"{dataset_id}" + "/{epoch:02d}-{val_loss:.4f}",
        monitor='train_loss',
        save_top_k=config.save_top_k,
        mode='min',)

    litmodel = LitNAM(config, model)

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(litmodel,
        train_dataloaders=train_loader, #val_dataloaders=valloader)
    )

    feat_outputs = model.calc_outputs(X_test.to('cpu'))
    feat_outputs = torch.cat(feat_outputs, dim=-1)
    # Testing the trained model


    test_info = trainer.test(dataloaders=test_loader)
    auroc = test_info[0]['roc_auc_epoch']
    accuracy = test_info[0]['Accuracy_metric_epoch']

    end_time = time.time()

    output_info = {
        'test_auroc': auroc,
        'test_accuracy': accuracy,
        'time': end_time - start_time,
    }
    with open(os.path.join(output_directory, 'output_info.json'), 'w') as f:
        json.dump(output_info, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--nr_epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--weight_norm",
        type=float,
        default=0,
        help="Weight norm",
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
        default=31,
        help='Dataset id',
    )
    parser.add_argument(
        '--test_split_size',
        type=float,
        default=0.2,
        help='Test size',
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
        default=False,
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
