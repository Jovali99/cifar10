import os
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import tensor, cat, save, load, optim, nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# --- Simple dataset wrapper for tensors ---
class TensorDataset(Dataset):
    def __init__(self, data, targets, mean=None, std=None):
        """
        Args:
            data (Tensor): Shape (N, C, H, W), values in [0,1]
            targets (Tensor): Shape (N,)
        """
        assert data.shape[0] == targets.shape[0], "Data and targets must have same length"
        assert data.max() <= 1.0 and data.min() >= 0.0, "Data must be in [0,1]"

        self.data = data
        self.targets = targets

        # Compute normalization stats if not provided
        if mean is None or std is None:
            self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
            self.std = self.data.std(dim=(0, 2, 3)).view(-1, 1, 1)
        else:
            self.mean = mean
            self.std = std

    def __getitem__(self, index):
        x = (self.data[index] - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.targets)

def loadDataset(data_cfg):
    dataset_name = data_cfg["dataset"]
    root = data_cfg["root"]

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
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(dataset, file)
            print(f"Population dataset saved to {file_path}")

def splitDataset(dataset, train_frac, test_frac):
    dataset_size = len(dataset)
    total = train_frac + test_frac
    assert np.isclose(total, 1.0), "Train + Test fractions must sum to 1.0"

    test_size = int(test_frac * dataset_size)

    indices = np.arange(dataset_size)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, shuffle=True)
    return train_idx, test_idx

def processDataset(data_cfg, trainset, testset):
    print("-- Processing dataset for training --")

    f_train = float(data_cfg["f_train"])
    f_test = float(data_cfg["f_test"]) 

    train_data, test_data, train_targets, test_targets = toTensor(trainset, testset)

    data = cat([train_data, test_data], dim=0)
    targets = cat([train_targets, test_targets], dim=0)

    assert len(data) == 60000, "Population dataset should contain 60000 samples"

    mean = data.mean(dim=(0, 2, 3))
    std = data.std(dim=(0, 2, 3))

    train_indices, test_indices = splitDataset(data, f_train, f_test)

    # --- Create normalized TensorDatasets ---
    train_dataset = TensorDataset(data[train_indices], targets[train_indices], mean, std)
    test_dataset = TensorDataset(data[test_indices], targets[test_indices], mean, std)

    # --- Assertion checks ---
    sample_x, sample_y = train_dataset[0]
    assert sample_x.shape == (3, 32, 32), f"Unexpected sample shape: {sample_x.shape}"
    assert not torch.isnan(sample_x).any(), "NaNs found in normalized data"
    assert not torch.isinf(sample_x).any(), "Infs found in normalized data"
    assert 0.0 <= sample_y < 10, f"Target out of range: {sample_y}"

    print(f"✅ Dataset ready | Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, test_dataset

# Used for optuna studies
def get_dataloaders(batch_size, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
