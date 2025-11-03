import os
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
        x = (self.data[index] - self.mean) / self.std
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.targets)

def loadDataset(data_cfg):
    dataset_name = data_cfg["dataset"]
    root = data_cfg["data_dir"]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
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

def processDataset(train_cfg, trainset, testset):
    print("-- Processing dataset for training & auditing  --")
    
    train_data, test_data, train_targets, test_targets = toTensor(trainset, testset)

    data = cat([train_data, test_data], dim=0)
    targets = cat([train_targets, test_targets], dim=0)

    assert len(data) == 60000, "Population dataset should contain 60000 samples"

    data_attrib = train_cfg["data"]
    train_attrib = train_cfg["train"]

    dataset_name = data_attrib["dataset"]
    file_path = os.path.join("data", f"{dataset_name}.pkl")
    saveDataset({"data": data, "targets": targets}, file_path)
    
    train_frac = data_attrib["f_train"]
    test_frac = data_attrib["f_test"]
    batch_size = train_attrib["batch_size"]

    print("-- Preparing dataset loaders --")
    train_indices, test_indices = splitDataset(data, train_frac, test_frac)
    train_loader, test_loader = prepareDataloaders(data, targets, train_indices, test_indices, batch_size)

    return train_loader, test_loader, train_indices, test_indices

def splitDataset(dataset, train_frac, test_frac):
    dataset_size = len(dataset)
    total = train_frac + test_frac
    assert np.isclose(total, 1.0), "Train + Test fractions must sum to 1.0"

    test_size = int(test_frac * dataset_size)

    indices = np.arange(dataset_size)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, shuffle=True)
    return train_idx, test_idx

def prepareDataloaders(data, targets, train_idx, test_idx, batch_size):
    full_dataset = TensorDataset(data, targets)

    # Create subsets sharing the same normalization stats
    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    print(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    return train_loader, test_loader
