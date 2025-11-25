from LeakPro.leakpro.attacks.mia_attacks.lira import lira_vectorized
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from src.save_load import saveShadowModelSignals, saveTargetSignals

import os
import numpy as np
import torch


def print_yaml(data, indent=0):
    """Recursively print YAML."""
    spacing = "    " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}")
            print_yaml(value, indent + 1)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            print(f"{spacing}- Item {index + 1}:")
            print_yaml(item, indent + 1)
    else:
        print(f"{spacing}{data}")

def bootstrap_sampling(K, M, shadow_logits, shadow_inmask, target_logits = None, target_inmask = None, replace = True, vec_mia_fun = lira_vectorized):
    noneflag = False 
    if target_logits is None:
        assert target_inmask is None
        noneflag = True
    elif target_logits.ndim == 1:
        assert target_inmask.ndim == 1
        target_logits = target_logits.reshape([-1,1])
        target_inmask = target_inmask.reshape([-1,1])
    else:
        assert target_logits.ndim == 2
        assert target_inmask.ndim == 2
    
    no_models = shadow_logits.shape[1] 
    ii_models = np.arange(no_models)
    results = []
    for m in range(M):
        if noneflag:
            i_target = np.random.randint(no_models)
            target_logits, target_inmask = shadow_logits[:,[i_target]], shadow_inmask[:,[i_target]]
            ii_remain = np.setdiff1d(ii_models,i_target)
        else:
            ii_remain = ii_models
        ii_sample = np.random.choice(ii_remain, K, replace)
        j = np.random.randint(target_logits.shape[1])
        #print(j, target_logits.shape, target_inmask.shape)
        score = vec_mia_fun(shadow_logits[:,ii_sample], shadow_inmask[:,ii_sample], target_logits[:,j])
        mask = ~np.isnan(score)
        #if not all(mask):
        #    print("number of nan scores:", np.isnan(score).sum())
        fpr, tpr, thresholds =  roc_curve(target_inmask[mask,j], score[mask])        
        results.append((fpr, tpr))
    return results

def interpolate_unique(fpr0, fpr, tpr, extrapolate=np.nan):
    # Sort fpr and tpr together
    sorted_indices = np.argsort(fpr)
    x = fpr[sorted_indices]
    y = tpr[sorted_indices]

    # Remove left duplicates
    filter_indices = np.append(x[:-1] != x[1:], True)
    x = x[filter_indices]
    y = y[filter_indices]

    return np.interp(fpr0, x, y, left=extrapolate, right=extrapolate)

def sigmoid_weigths(score: np.ndarray, centrality: float, temperature: float) -> np.ndarray:
    exp = np.exp((score-centrality)/temperature)
    weight = 1.0/(1.0+exp)
    return weight

def calculate_logits(model, dataset, device, batch_size=128) -> np.ndarray:
    model.eval()
    logits_list = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.to(device)
            out = model(x)      # logits
            logits_list.append(out.cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    return logits

def percentile_score_normalization(scores: np.ndarray, percentile: int, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize scores using percentile clipping.
    
    Args:
        scores: Array of base model audit scores.
        percentile: Percentile to clip extremes (e.g., 2 will clip lower 2% and upper 98%).
        eps: Small number to prevent division by zero.
        
    Returns:
        Normalized scores in [0, 1].
    """
    if(percentile > 50):
        percentile = 100 - percentile
    lo = np.percentile(scores, percentile)
    hi = np.percentile(scores, 100-percentile)
    norm = (scores - lo) / (hi - lo + eps)
    return np.clip(norm, 0.0, 1.0)

def print_percentiles(threshold, scores):
    """
    prints the percentiles from a set threshold

    args:
        threshold: Value threshold to print percentiles outside it
        scores: The scores, either normalized or normal
    """
    threshold = 5  # change this to whatever you want
    num_above = np.sum(scores > threshold)
    print(f"Number of scores above {threshold}: {num_above}")

    percent_above = 100 * num_above / len(scores)
    print(f"Percentage of scores above {threshold}: {percent_above:.2f}%")

    threshold_2 = -threshold  # change this to whatever you want
    num_below = np.sum(scores < threshold_2)
    print(f"Number of scores below {threshold_2}: {num_below}")

    percent_below = 100 * num_below / len(scores)
    print(f"Percentage of scores above {threshold_2}: {percent_below:.2f}%")

def get_shadow_signals(shadow_logits, shadow_inmask, amount):
    total_models = shadow_logits.shape[1]

    if amount > total_models:
        raise ValueError(f"Requested {amount} shadow models but only {total_models} available.")

    # Choose random shadow model indices
    selected_indices = np.random.choice(total_models, size=amount, replace=False)

    # Select those columns
    logits_sub = shadow_logits[:, selected_indices]
    inmask_sub = shadow_inmask[:, selected_indices]
    return logits_sub, inmask_sub

def rescale_logits(logits, true_label):
    if logits.shape[1] == 1:
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        positive_class_prob = sigmoid(logits).reshape(-1, 1)
        predictions = np.concatenate([1 - positive_class_prob, positive_class_prob], axis=1)
    else:
        predictions = logits - np.max(logits, axis=1, keepdims=True)
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)

    count = predictions.shape[0]
    y_true = predictions[np.arange(count), true_label]
    predictions[np.arange(count), true_label] = 0
    y_wrong = np.sum(predictions, axis=1)

    output_signals = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    return output_signals

def calculate_logits_and_inmask(dataset, model, metadata, path, idx: int | None = None):
    """
    Calculates logits and the inmask for a given model. If an index is input
    the function assumes the model is a shadow model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(idx is not None):
        print(f"\n=== Calculating logits and in_mask for shadow model {idx} ===")
    else:
        print(f"\n=== Calculating logits and in_mask for target model ===")

    model = model.to(device)

    logits = calculate_logits(model, dataset, device)
    
    # Create the in_mask from training indices
    in_mask = np.zeros(len(dataset), dtype=np.bool_)

    in_indices = np.array(metadata.train_indices, dtype=np.int64)
    in_mask[in_indices] = True
    if(idx is not None):
        saveShadowModelSignals(logits, in_mask, idx, path)
    else:
        print(f"Logits shape: {logits.shape}")
        print(f"In_mask shape: {in_mask.shape}")
        saveTargetSignals(logits, in_mask, path)

    del logits
    del in_mask
