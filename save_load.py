import hashlib
import os
import numpy as np
import json

def buildMetadata(trainCfg: dict, auditCfg: dict = {}, additionalCfg: dict = {}) -> dict:
    """
    Construct metadata describing the training configuration, hyperparameters, 
    and (optionally) audit configuration.
    """
    metadata = {
        "trainCfg": trainCfg,
        "auditCfg": auditCfg if auditCfg is not None else {},
        "additionalCfg": additionalCfg if additionalCfg is not None else {}
    }
    return metadata

def hashCfg(metadata:dict, inmask: np.ndarray) -> str:
    """
    Compute a unique SHA256 hash based on metadata and inmask.
    """
    hash = hashlib.sha256()

    # Hash metadata
    meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    hash.update(meta_bytes)

    # Hash inmask
    hash.update(inmask.tobytes())

    return hash.hexdigest()

def save(metadata: dict, model_logits: np.ndarray, inmask: np.ndarray = None, savePath:str = "saved_models", ):
    """
    Save model logits, inmask, and metadata into a folder named by its unique hash.
    """
    os.makedirs(savePath, exist_ok=True)

    hash_id = hashCfg(metadata, inmask)
    # Appends 0- to the hash id as index of model, this is to be changed when
    # shadow models are going to be trained
    save_dir = os.path.join(savePath, "0-"+hash_id)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "model_logits.npy"), model_logits)

    if inmask is not None:
        np.save(os.path.join(save_dir, "inmask.npy"), inmask)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    print(f"✅ Saved model and inmask with hash_id: {hash_id}")
    return hash_id, save_dir
