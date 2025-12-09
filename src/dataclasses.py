from dataclasses import dataclass
import numpy as np

@dataclass
class FbdTrialResults:
    accuracy: float
    noise: float
    centrality: float
    temperature: float
    tau: float  # tau@0.1

@dataclass
class FbdArgs:
    rmia_scores: np.ndarray
    train_dataset: object
    test_dataset: object
    shadow_gtl_probs: np.ndarray
    shadow_inmask: np.ndarray
    target_inmask: np.ndarray
    tauc_ref: float

class CIFARDatasetStructure:
    """Holds .data (N,H,W,3) and .targets just like CIFAR10."""
    def __init__(self, data, targets):
        self.data = data  # numpy array uint8
        self.targets = targets  # list of ints

    def __len__(self):
        return len(self.targets)
