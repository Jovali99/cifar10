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