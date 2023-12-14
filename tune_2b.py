import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import random
import numpy as np
import argparse

import  yaml
from easydict import EasyDict
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from dataloader.bci_compet import get_dataset
from dataloader.bci_compet import BCICompet2bIV

from model.litmodel import LitModel
from model.litmodel import get_litmodel

from model.cat_conditioned import CatConditioned
from model.attn_conditioned import ATTNConditioned
from model.attn_conditioned_subj_avg import ATTNConditionedSubjAvg
from model.attn_conditioned_subj_ftr import ATTNConditionedSubjFtr


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything


from utils.setup_utils import (
    get_device,
    get_log_name,
)
from utils.training_utils import get_callbacks


import optuna
from functools import partial

torch.set_float32_matmul_precision('medium')

CACHE_ROOT = 'cache'
config_name = 'bcicompet2b_config'
METHODS = ["baseline", "cat", "attn", "avg", "ftr"]
info_lookup = {"baseline": False, "cat": "id", "attn": "id", "avg": "avg", "ftr": "ftr"}
train_size = 240
val_size = 48

def parse_args(args):
    parser = argparse.ArgumentParser(description="Bayesn hypertune.")
    parser.add_argument(
        "method",
        type=str,
        help="baseline OR cat OR attn OR avg OR ftr",
    )
    parser.add_argument("--n_trials", type=int, default = 2, help="Number of trials to hypertune.")
    args = parser.parse_args(args)
    assert args.method in METHODS, "Invalid method"
    return args


def get_dataset(args):
    def load_dataset(args):
        datasets = {}
        for subject_id in range(0,9):
            args['target_subject'] = subject_id
            datasets[subject_id] = BCICompet2bIV(args)
        return datasets

    path = os.path.join(CACHE_ROOT, f'{config_name}.pkl')

    if not os.path.isfile(path):
        print('Cache miss, generating cache')
        datasets = load_dataset(args)
        with open(path, 'wb') as file:
            pickle.dump(datasets, file)
    else:
        print('Loading cache')
        with open(path, 'rb') as file:
            datasets = pickle.load(file)

    for subject_id in datasets.keys(): 
        datasets[subject_id].return_subject_info = args['return_subject_info']
    return datasets


def train_fnc(trial, args, train_dataloader, test_dataloader):
    args['TUNE_VERSION'] = args['VERSION'] + f'-{trial.number}'
    # Hyperparameters to be tuned by Optuna.
    hyperparams = {
        "lr": 3.0e-4, # trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        #"epochs": trial.suggest_int("epochs", 10, 1000),
        "epochs": args['EPOCHS'],
        "dropout_probability": trial.suggest_float("dropout", 0.1, 0.4),
        #'dropout_probability': 0.2,
        "eeg_normalization": "LayerNorm", #trial.suggest_categorical("eeg_norm", ["None", "CondBatchNorm", "LayerNorm"]),
        "subject_normalization": "LayerNorm", #trial.suggest_categorical("subj_norm", ["None", "CondBatchNorm", "LayerNorm"]),
        "eeg_activation": False, # trial.suggest_categorical("eeg_activation", [True, False]),
        "embedding_dimension": trial.suggest_int("embedding_dimension", 2, 64),
        "combined_features_dimension": trial.suggest_int("combined_features_dimension", 32, 128),
    }
    # if hyperparams["combined_features_dimension"]  > 9:
    #     hyperparams["combined_features_dimension"] = None

    if args.method  == "ftr":
        hyperparams["subj_dim"] = trial.suggest_int("subj_dim", 4, 32)

    args['lr'] = hyperparams['lr']

    if args.method == "baseline":
        hyperparams = {
            "dropout_probability": trial.suggest_float("dropout", 0.0, 0.5),
        }
        args['dropout_rate'] = hyperparams['dropout_probability']
        hyperparams['epochs'] = args['EPOCHS']
  

    if args.method == "baseline":
        lit_model = get_litmodel(args)
    elif args.method == "cat":
        model = CatConditioned(args, eeg_normalization=hyperparams["eeg_normalization"], eeg_activation=hyperparams["eeg_activation"], embedding_dimension=hyperparams["embedding_dimension"], subject_normalization=hyperparams["subject_normalization"], dropout_probability=hyperparams["dropout_probability"], combined_features_dimension=hyperparams["combined_features_dimension"] )
        lit_model = LitModel(args, model)
    elif args.method == "attn":
        model = ATTNConditioned(args, eeg_normalization=hyperparams["eeg_normalization"], eeg_activation=hyperparams["eeg_activation"], embedding_dimension=hyperparams["embedding_dimension"], subject_normalization=hyperparams["subject_normalization"], dropout_probability=hyperparams["dropout_probability"], combined_features_dimension=hyperparams["combined_features_dimension"])
        lit_model = LitModel(args, model)
    elif args.method == "avg":
        model = ATTNConditionedSubjAvg(args, eeg_normalization=hyperparams["eeg_normalization"], eeg_activation=hyperparams["eeg_activation"], embedding_dimension=hyperparams["embedding_dimension"], subject_normalization=hyperparams["subject_normalization"], dropout_probability=hyperparams["dropout_probability"], combined_features_dimension=hyperparams["combined_features_dimension"] )
        lit_model = LitModel(args, model)
    elif args.method == "ftr":
        model = ATTNConditionedSubjFtr(args,  eeg_normalization=hyperparams["eeg_normalization"], eeg_activation=hyperparams["eeg_activation"], subj_dim=hyperparams["subj_dim"], embedding_dimension=hyperparams["embedding_dimension"], subject_normalization=hyperparams["subject_normalization"], dropout_probability=hyperparams["dropout_probability"], combined_features_dimension=hyperparams["combined_features_dimension"] )
        lit_model = LitModel(args, model)
    else:
        throw("Unknown method")

    logger = TensorBoardLogger(args.LOG_PATH, 
                                    name=args.VERSION)
    callbacks = get_callbacks(monitor='val_loss', args=args)

    trainer = Trainer(
            max_epochs=hyperparams['epochs'],
            callbacks=callbacks,
            default_root_dir=args.CKPT_PATH,
            logger=logger,
            enable_progress_bar=False
        )
    trainer.fit(lit_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader)
        
    torch.cuda.empty_cache()
    last_val_acc = lit_model.val_accs[-1]
    best_val_acc = max(lit_model.val_accs)
    print("Last val acc: ", last_val_acc, ", Best val acc: ", best_val_acc)
    return best_val_acc

def main(cmd_args=None):
    os.system('clear')
    ### Load Config ###
    cmd_args = parse_args(cmd_args)
    with open(f'configs/{config_name}.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        args = EasyDict(config)
    args['method'] = cmd_args.method
    args['return_subject_info'] = info_lookup[cmd_args.method]
    args['VERSION'] = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{cmd_args.method}'
   
    
    
    #### Set Log ####
    args['current_time'] = datetime.now().strftime('%Y%m%d')
    args['LOG_NAME'] = get_log_name(args)

    #### Update configs ####
    if args.downsampling != 0: args['sampling_rate'] = args.downsampling
    seed_everything(args.SEED)

   

    ### Load Dataset ###
    datasets = get_dataset(args)

    train_datasets = {}
    val_datasets = {}
    for subject_id in datasets.keys():
        train_datasets[subject_id] = torch.utils.data.Subset(datasets[subject_id], range(train_size))
        val_datasets[subject_id] = torch.utils.data.Subset(datasets[subject_id], range(train_size, train_size+val_size))


    train_dataset_all = torch.utils.data.ConcatDataset(list(train_datasets.values()))
    val_dataset_all = torch.utils.data.ConcatDataset(list(val_datasets.values()))
    print("Train: ", len(train_dataset_all), ", Test: ", len(val_dataset_all))

    train_dataloader_all = DataLoader(train_dataset_all, batch_size=args['batch_size'], shuffle=True, num_workers=0, persistent_workers=False)
    val_dataloader_all = DataLoader(val_dataset_all, batch_size=args['batch_size'], shuffle=False, num_workers=0, persistent_workers=False)

    print("Starting hypertune! Method: ", args.method)
    #input("Press Enter to continue...")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3", 
        study_name=args["VERSION"],
        load_if_exists=True,
    )
    objective_partial = partial(train_fnc, args=args, train_dataloader=train_dataloader_all, test_dataloader=val_dataloader_all)
    study.optimize(objective_partial, n_trials=cmd_args.n_trials)

    print()
    print("Best Hyperparameters:", study.best_params)

if __name__ == "__main__":
    main()
