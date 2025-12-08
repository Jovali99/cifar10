from src.cifar_handler import CifarInputHandler
from src.dataset_handler import get_dataloaders, get_weighted_dataloaders
from src.models.resnet18_model import ResNet18
from src.models.wideresnet28_model import WideResNet
from src.utils import sigmoid_weigths, calculate_logits, rescale_logits, calculate_tauc
from src.save_load import saveTrial, buildTrialMetadata
from torch import nn, optim
from LeakPro.leakpro.attacks.mia_attacks.rmia import rmia_get_gtlprobs, rmia_vectorised
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import optuna
import os

# Define the datasets
train_dataset = None
test_dataset = None

DEVICE = None

def train_one_epoch(model, optimizer, train_loader, device, epoch, epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total

def objective(trial, config):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    T_max = trial.suggest_int("T_max", 20, 50)

    train_loader, val_loader = get_dataloaders(batch_size, train_dataset, test_dataset)

    if config["data"]["dataset"] == "cifar10" or config["data"]["dataset"] == "cinic10":
        n_classes = 10
    elif config["data"]["dataset"] == "cifar100":
        n_classes = 100
    else:
        raise ValueError(f"Incorrect dataset {config['data']['dataset']}")

    if config["study"]["model"] == "resnet":
        model = torchvision.models.resnet18(num_classes=n_classes).to(DEVICE)
        print("Optimizing resnet")
    elif config["study"]["model"] == "wideresnet":
        drop_rate = trial.suggest_float("drop_rate", 0.0, 0.5)
        model = WideResNet(depth=28, num_classes=n_classes, widen_factor=10, dropRate=drop_rate).to(DEVICE)
        print("Optimizing wideresnet")
        
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    max_epochs = 50
    best_val_accuracy = 0.0
    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, max_epochs)
        scheduler.step()
        val_accuracy = evaluate(model, val_loader, DEVICE)
        print(f"Trial val accuracy: {val_accuracy}")
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy

    return best_val_accuracy

def fbd_objective(trial, cfg, rmia_scores, train_dataset, test_dataset, shadow_gtl_probs, shadow_inmask, target_inmask, tauc_ref, save_path, device):
    """
        noise_std: Trial between [0.001, 0.1]
        Centrality: Trial stepped between [0.0, 1.0]
        Temperature: Trial between [0.05, 0.5]
    """
    # study params
    noise_std = trial.suggest_float("noise_std", 1e-3, 1e-1, log=True)
    centrality = trial.suggest_float("centrality", 0.0, 1.0, step=0.1)
    temperature = trial.suggest_float("temperature", 5e-2, 5e-1, step=0.05)

    # Calculate the weights
    weights = sigmoid_weigths(rmia_scores, centrality, temperature)

    lr = cfg["fbd_study"]["learning_rate"]
    weight_decay = cfg["fbd_study"]["weight_decay"]
    epochs = cfg["fbd_study"]["epochs"]
    momentum = cfg["fbd_study"]["momentum"]
    t_max = cfg["fbd_study"]["t_max"]
    batch_size = cfg["fbd_study"]["batch_size"]

    if(cfg["data"]["dataset"] == "cifar10" or cfg["data"]["dataset"] == "cinic10"):
        num_classes = 10
    elif(cfg["data"]["dataset"] == "cifar100"):
        num_classes = 100
    else:
        raise ValueError(f"Incorrect dataset {cfg['data']['dataset']}")

    model = ResNet18(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    train_loader, test_loader = get_weighted_dataloaders(batch_size, train_dataset, test_dataset, weights)

    handler = CifarInputHandler();
    handler.trainStudyFbD(train_loader, model, criterion, optimizer, epochs, noise_std, scheduler)
    test_accuracy = handler.eval(test_loader, model, criterion).accuracy

    # ----------------- GÖR DETTA I TRAIN_MODELS -----------------

    assert train_dataset.dataset is test_dataset.dataset, "train_dataset.dataset =/= test_dataset.dataset"
    full_dataset = train_dataset.dataset

    model.to(device)
    target_logits = calculate_logits(model, full_dataset, device)
    labels = np.array(full_dataset.targets)

    rescaled_target_logits = rescale_logits(target_logits, labels)
    target_gtl_probs = rmia_get_gtlprobs(target_logits, labels)
    
    scores = rmia_vectorised(target_gtl_probs, shadow_gtl_probs, shadow_inmask, online=True, use_gpu_if_available=True)

    model.to("cpu")

    tauc_weighted = calculate_tauc(scores, target_inmask, fpr=0.1)
    tau = np.log(tauc_weighted/tauc_ref)

    metadata = buildTrialMetadata(noise_std, centrality, temperature, test_accuracy, tau)
    saveTrial(metadata, target_gtl_probs, rescaled_target_logits, trial.number, save_path)

    return tau, test_accuracy


