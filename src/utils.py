from LeakPro.leakpro.attacks.mia_attacks.lira import lira_vectorized
from sklearn.metrics import roc_curve
import numpy as np

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

