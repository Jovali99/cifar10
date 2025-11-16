import hashlib
import os
import numpy as np
import json
from datetime import datetime

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
              audit_data_indices: np.ndarray, savePath:str = "audit_signals"):
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
