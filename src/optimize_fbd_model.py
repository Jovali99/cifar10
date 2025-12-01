from dataclasses import dataclass
import optuna
import pandas as pd
import numpy as np
import yaml
import pickle
import torch
import os
import matplotlib.pyplot as plt
import multiprocessing

import src.save_load as sl

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import tensor, cat, save, load, optim, nn
from torch.utils.data import DataLoader
from src.models.resnet18_model import ResNet18

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import src.study_handler as sh
from src.utils import print_yaml, get_shadow_signals, calculate_tauc
from LeakPro.leakpro.attacks.mia_attacks.rmia import rmia_vectorised, rmia_get_gtlprobs
from src.save_load import loadTargetSignals, loadShadowModelSignals

@dataclass
class FbdArgs:
    rmia_scores: np.ndarray
    train_dataset: object
    test_dataset: object
    shadow_gtl_probs: np.ndarray
    shadow_inmask: np.ndarray
    target_inmask: np.ndarray
    tauc_ref: float

multiprocessing.set_start_method('spawn')

def run_optimization(config, gpu_id, trials, save_path, fbd_args: FbdArgs): 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    study_cfg = config['fbd_study']

    # Parallell storage setup
    db_path = os.path.join(study_cfg['root'], "fbd_study.db")
    storage = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=study_cfg["study_name"],
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "maximize"]
    )
    # Extract arguments from dataclass
    rmia_scores = fbd_args.rmia_scores
    train_dataset = fbd_args.train_dataset
    test_dataset = fbd_args.test_dataset
    shadow_gtl_probs = fbd_args.shadow_gtl_probs
    shadow_inmask = fbd_args.shadow_inmask
    target_inmask = fbd_args.target_inmask
    tauc_ref = fbd_args.tauc_ref
    
    func = lambda trial: sh.fbd_objective(trial, config, rmia_scores, train_dataset, 
                                          test_dataset, shadow_gtl_probs, shadow_inmask, 
                                          target_inmask, tauc_ref, gpu_id, save_path)
    
    study.optimize(func, n_trials=trials)
    
    print(f"Study '{study_cfg['study_name']}' completed on GPU {gpu_id}.")
    df = study.trials_dataframe() 
    df.to_csv(os.path.join(save_path, f"results_gpu_{gpu_id}.csv"), index=False) 
    print(f"📄 Results saved to {os.path.join(save_path, f'results_gpu_{gpu_id}.csv')}")

def parallell_optimization(config, labels, gpu_ids, fbd_args):
    study_cfg = config['fbd_study'] 

    metadata = sl.buildStudyMetadata(study_cfg, config['data']) 
    _, save_path = sl.saveStudy(metadata, savePath=study_cfg['root'], labels=labels) 

    assert study_cfg["trials"] % len(gpu_ids) == 0, f"amount of trials {study_cfg['trials']} cannot be equally split among {len(gpu_ids)}"
    trials = study_cfg["trials"] // len(gpu_ids)
    
    processes = [multiprocessing.Process(
        target=run_optimization, 
        args=(config, gpu_id, trials, save_path, fbd_args)
    ) for gpu_id in gpu_ids] 
    
    for p in processes:
        p.start() 
    for p in processes:
        p.join()
        
    db_path = os.path.join(study_cfg['root'], "fbd_study.db")
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_cfg["study_name"], storage=storage)
    return study