import hashlib
import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

def buildAuditMetadata(trainCfg: dict, auditCfg: dict = {}) -> dict:
    """
    Construct metadata describing the training configuration, hyperparameters, 
    and (optionally) audit configuration.
    """
    metadata = {
        "trainCfg": trainCfg,
        "auditCfg": auditCfg
    }
    return metadata

def buildTargetMetadata(trainCfg: dict, dataCfg: dict, additionalCfg: dict = {}) -> dict:
    """
    Construct metadata describing the study configuration. 
    """
    metadata = {
        "train": trainCfg,
        "data": dataCfg,
        "additionalCfg": additionalCfg if additionalCfg is not None else {}
    }
    return metadata

def buildStudyMetadata(studyCfg: dict, dataCfg: dict, additionalCfg: dict = {}) -> dict:
    """
    Construct metadata describing the study configuration. 
    """
    metadata = {
        "study": studyCfg,
        "data": dataCfg,
        "additionalCfg": additionalCfg if additionalCfg is not None else {}
    }
    return metadata

def hashCfg(metadata:dict, inmask: np.ndarray = None) -> str:
    """
    Compute a unique SHA256 hash based on metadata and inmask.
    """
    hash = hashlib.sha256()

    # Hash metadata
    meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    hash.update(meta_bytes)

    # Hash inmask
    if inmask is not None:
        hash.update(inmask.tobytes())

    return hash.hexdigest()[:10]

def saveAudit(metadata: dict, target_model_logits: np.ndarray,
              shadow_models_logits: np.ndarray, inmask: np.ndarray,
              target_inmask: np.ndarray, audit_data_indices: np.ndarray, savePath:str = "audit_signals"):
    """
    Save a full audit signals into a folder named by its metadata unique hash.
        metadata: the training and audit configuration
        target_logits: Rescaled target model logits
        shadow_models_logits: Rescaled shadow model logits
        inmask: in_indices_mask for the shadow models
        audit_data_indices: indices used for audit
    """
    os.makedirs(savePath, exist_ok=True)

    hash_id = hashCfg(metadata, inmask)

    date_str = datetime.now().strftime("%Y%m%d")

    folder_name = f"{date_str}_{hash_id}"

    save_dir = os.path.join(savePath, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "rescaled_target_logits.npy"), target_model_logits)
    np.save(os.path.join(save_dir, "rescaled_shadow_model_logits.npy"), shadow_models_logits)
    np.save(os.path.join(save_dir, "shadow_models_in_mask.npy"), inmask)
    np.save(os.path.join(save_dir, "target_in_mask.npy"), inmask)
    np.save(os.path.join(save_dir, "audit_data_indices.npy"), audit_data_indices)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    print(f"✅ Saved target and shadow models logits, inmask and audit data indices with hash_id: {hash_id}")
    return hash_id, save_dir

def saveStudy(metadata: dict, savePath:str = "study"):
    """
    Create a uniquely hashed folder for each Optuna study based on its metadata.
    Save metadata.json in that folder and return (hash_id, save_dir).
    """
    os.makedirs(savePath, exist_ok=True)

    hash_id = hashCfg(metadata)
    studyCfg = metadata['study']
    study_name = studyCfg['study_name']

    # Construct save directory
    save_dir = os.path.join(savePath, f'{study_name}-{hash_id}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    print(f"✅ Saved study journal and study metadata with hash_id: {hash_id}")
    return hash_id, save_dir

def saveTarget(metadata: dict, savePath:str = "target"):
    """
    Create a uniquely hashed folder for each target training based on its metadata.
    Save metadata.json in that folder and return (hash_id, save_dir).
    """
    os.makedirs(savePath, exist_ok=True)

    hash_id = hashCfg(metadata)

    # Construct save directory
    save_dir = os.path.join(savePath, f'{"resnet18"}-{hash_id}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    print(f"✅ Saved training metadata with hash_id: {hash_id}")
    return hash_id, save_dir

def saveTargetSignals(target_model_logits: np.ndarray, in_mask: np.ndarray, path):
    """
    Saves the logits and in_mask of a target model
    """
    # Save logits
    save_logits_path = os.path.join(path, "target_logits.npy")
    np.save(save_logits_path, target_model_logits)

    # Save in_mask
    save_in_mask_path = os.path.join(path, f"target_in_mask.npy")
    np.save(save_in_mask_path, in_mask)
    print(f"✅ Saved target model logits at: {save_logits_path}, and in mask at {save_in_mask_path}")

def saveShadowModelSignals(logits: np.ndarray, in_mask, identifier: int, path: str = "processed_shadow_models"):
    """
    Saves the logits and in_mask of a shadow model
    """
    # Save logits
    save_logits_path = os.path.join(path, f"shadow_logits_{identifier}.npy")
    np.save(save_logits_path, logits)
    
    # Save in_mask
    save_in_mask_path = os.path.join(path, f"in_mask_{identifier}.npy")
    np.save(save_in_mask_path, in_mask)

    print(f"✅ Saved shadow model logits at: {save_logits_path}, and in mask at {save_in_mask_path}")

def loadAudit(audit_signals_name: str, save_path: str = "audit_signals"):
    """
    Load audit data previously saved with saveAudit().
    
    audit_signals_name:
        Folder name of the audit run (e.g. '20250110_123456789')
    save_path:
        Base folder where audit runs are stored.
    
    Returns:
        metadata (dict)
        rescaled_target_logits (np.ndarray)
        rescaled_shadow_model_logits (np.ndarray)
        shadow_models_in_mask (np.ndarray)
        audit_data_indices (np.ndarray)
    """
    audit_dir = os.path.join(save_path, audit_signals_name)

    if not os.path.exists(audit_dir):
        raise FileNotFoundError(f"Audit directory not found: {audit_dir}")

    # --- Load files ---
    metadata_path = os.path.join(audit_dir, "metadata.json")
    target_logits_path = os.path.join(audit_dir, "rescaled_target_logits.npy")
    shadow_logits_path = os.path.join(audit_dir, "rescaled_shadow_model_logits.npy")
    inmask_path = os.path.join(audit_dir, "shadow_models_in_mask.npy")
    target_inmask_path = os.path.join(audit_dir, "target_in_mask.npy")
    indices_path = os.path.join(audit_dir, "audit_data_indices.npy")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load numpy arrays
    rescaled_target_logits = np.load(target_logits_path)
    rescaled_shadow_model_logits = np.load(shadow_logits_path)
    shadow_models_in_mask = np.load(inmask_path)
    target_in_mask = np.load(target_inmask_path)
    audit_data_indices = np.load(indices_path)

    print(f"📥 Loaded audit signals from folder: {audit_signals_name}")

    return (metadata, rescaled_target_logits, rescaled_shadow_model_logits,
            shadow_models_in_mask, target_in_mask, audit_data_indices)

def savePlot(fig, filename: str, audit_dir: str, savePath: str = "audit_signals", dpi: int = 300, fmt: str = "png"):
    """
    Save a matplotlib figure with high-quality settings.
    
    Parameters:
        fig: matplotlib.figure.Figure
            The figure object to save.
        audit_dir: str
            Name of the audit_signals subdir used for the creation of the plots
        filename: str
            Name of the file without extension.
        savePath: str
            Directory where the figure is stored.
        dpi: int
            Resolution for the exported image.
        fmt: str
            File format ("png", "pdf", "svg", ...).
    """
    plot_path = os.path.join(audit_dir, "plot")
    save_dir = os.path.join(savePath, plot_path)
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, f"{filename}.{fmt}")

    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"📁 Saved plot to: {full_path}")
