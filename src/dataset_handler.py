import os
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import tensor, cat, save, load, optim, nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from src.cifar_handler import CifarInputHandler

# Basic dataset class to handle weighted datsets
class weightedDataset(Dataset):
    def __init__(self, dataset, indices, weights):
        self.dataset: Dataset = dataset
        self.weights = weights
        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return self.indices[idx], self.weights[idx], data, label

def loadDataset(data_cfg):
    dataset_name = data_cfg["dataset"]
    root = data_cfg.get("root", data_cfg.get("data_dir"))
    
    transform = transforms.Compose([
       transforms.ToTensor(),  # Convert PIL image to Tensor
    ])

    trainset, testset = None, None
    if(dataset_name == "cifar10"):
        trainset = CIFAR10(root=root, train=True, download=True, transform=transform)
        testset = CIFAR10(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    assert trainset != None, "Failed loading the train set"
    assert testset != None, "Failed loading the test set"
    print("-- Dataset loaded: ", dataset_name, " --")
    return trainset, testset

def toTensor(trainset, testset):
    train_data = tensor(trainset.data).permute(0, 3, 1, 2).float() / 255
    test_data = tensor(testset.data).permute(0, 3, 1, 2).float() / 255

    train_targets = tensor(trainset.targets)
    test_targets = tensor(testset.targets)

    return train_data, test_data, train_targets, test_targets

def saveDataset(dataset, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)
        print(f"Dataset saved to {file_path}")

def splitDataset(dataset, train_frac, test_frac):
    dataset_size = len(dataset)
    total = train_frac + test_frac
    assert np.isclose(total, 1.0), "Train + Test fractions must sum to 1.0"

    test_size = int(test_frac * dataset_size)

    indices = np.arange(dataset_size)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, shuffle=True)
    return train_idx, test_idx

def processDataset(data_cfg, trainset, testset, in_indices_mask=None):
    print("-- Processing dataset for training --")

    f_train = float(data_cfg["f_train"])
    f_test = float(data_cfg["f_test"]) 

    train_data, test_data, train_targets, test_targets = toTensor(trainset, testset)

    data = cat([train_data.clone().detach(), test_data.clone().detach()], dim=0)
    targets = cat([train_targets, test_targets], dim=0)
    assert len(data) == 60000, "Population dataset should contain 60000 samples"

    dataset = CifarInputHandler.UserDataset(data, targets)

    # ---------------------------------------------------------------------
    # CASE 1 — Custom train indices given 
    # ---------------------------------------------------------------------
    if in_indices_mask is not None:
        print("Using provided in_indices_mask for training.")

        # Expected sizes (rounded)
        expected_train = int(f_train * len(dataset))
        expected_test = int(f_test * len(dataset))

        # Convert boolean mask → integer array of indices
        assert len(in_indices_mask) == len(dataset), \
            f"in_indices_mask has wrong length: {len(in_indices_mask)} but dataset has {len(dataset)}"

        # Ensure mask is boolean
        assert in_indices_mask.dtype == bool or set(np.unique(in_indices_mask)).issubset({0, 1}), "in_indices_mask must be boolean or contain only 0/1"

        # Extract the actual index positions
        train_indices = np.where(in_indices_mask == 1)[0]

        # Compute test indices = all remaining indices
        all_indices = np.arange(len(dataset))
        test_indices = np.setdiff1d(all_indices, train_indices, assume_unique=False)

        assert len(train_indices) == expected_train, f"Train size mismatch: mask gives {len(train_indices)} but expected {expected_train}"
        assert len(test_indices) == expected_test, f"Test size mismatch: mask gives {len(test_indices)} but expected {expected_test}"
    # ---------------------------------------------------------------------
    # CASE 2 — No custom indices
    # ---------------------------------------------------------------------
    else:
        train_indices, test_indices = splitDataset(dataset, f_train, f_test)

    # Save dataset
    dataset_name = data_cfg["dataset"]
    dataset_root = data_cfg.get("root", data_cfg.get("data_dir"))
    file_path = os.path.join(dataset_root, dataset_name + ".pkl")
    saveDataset(dataset, file_path)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # --- Assertion checks ---
    sample_x, sample_y = train_dataset[0]
    assert sample_x.shape == (3, 32, 32), f"Unexpected sample shape: {sample_x.shape}"
    assert not torch.isnan(sample_x).any(), "NaNs found in normalized data"
    assert not torch.isinf(sample_x).any(), "Infs found in normalized data"
    assert 0.0 <= sample_y < 10, f"Target out of range: {sample_y}"

    print(f"✅ Dataset ready | Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, test_dataset, train_indices, test_indices

# Used for optuna studies
def get_dataloaders(batch_size, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_weighted_dataloaders(batch_size, train_dataset, test_dataset, weights):
    train_indices = train_dataset.indices
    test_indices = test_dataset.indices

    train_weights = weights[train_indices]
    test_weights = weights[test_indices]

    weighted_train = weightedDataset(train_dataset, train_indices, train_weights)
    weighted_test = weightedDataset(test_dataset, test_indices, test_weights)

    train_loader = DataLoader(weighted_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(weighted_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
