import hashlib
import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from src.optimize_fbd_model import FbdTrialResults

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

def buildTrialMetadata(noise, centrality, temperature, accuracy, tau):
    metadata = {
        "noise": noise,
        "centrality": centrality,
        "temperature": temperature,
        "accuracy": accuracy,
        "tau@0.1": tau,
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

def saveStudy(metadata: dict, savePath:str = "study", labels: np.ndarray = None):
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
        
    if labels is not None:
        trial_outputs_dir = os.path.join(save_dir, "trial_outputs")
        os.makedirs(trial_outputs_dir, exist_ok=True)
        # Save the labels
        save_labels = os.path.join(trial_outputs_dir, f"labels.npy")
        np.save(save_labels, labels)

    print(f"✅ Saved study journal and study metadata with hash_id: {hash_id}")
    return hash_id, save_dir

def saveTrial(metadata: dict, gtl: np.ndarray, resc_logits: np.ndarray, idx: int, path: str):
    """Saves computed rescaled logits, gtl_probs for the weighted
       target model along with the resulting outputs and the used parameters

    Args:
        metadata (dict): Metadata containing parameters and evaluation outputs
        gtl (np.ndarray): gtl_probabilities used for rmia
        resc_logits (np.ndarray): rescaled_logits used for lira
        idx (int): trial index
        path (str): study/{study_folder}/...

    Returns:
        Confirmation and results
    """
    # Make sure the dir is created if it doesnt exist
    save_dir = os.path.join(path, "trial_outputs")
    
    # Save the metadata to trial_outputs/metadata/...
    md_dir = os.path.join(save_dir, "metadata")
    os.makedirs(md_dir, exist_ok=True)
    with open(os.path.join(md_dir, f"metadata_{idx}.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    # Save the rescaled logits to trial_outputs/rescaled_logits/...
    resc_logits_dir = os.path.join(save_dir, "rescaled_logits")
    os.makedirs(resc_logits_dir, exist_ok=True)
    np.save(os.path.join(resc_logits_dir, f"rescaled_logits_{idx}.npy"), resc_logits)
    
    # Save the gtl probabilities to trial_outputs/gtl_probabilities/...
    gtl_probs_dir = os.path.join(save_dir, "gtl_probabilities")
    os.makedirs(gtl_probs_dir, exist_ok=True)
    np.save(os.path.join(gtl_probs_dir, f"gtl_probabilities_{idx}.npy"), gtl)
    
    return print(f"✅ Saved trial #:{idx} logits and metadata with accuracy {metadata['accuracy']} and tau@0.1 {metadata['tau@0.1']} ")

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

def loadTargetSignals(target_name: str, path: str = "target"):
    """
    Loads the target logits, in_mask and metadata from input path
    """
    target_dir = os.path.join(path, target_name)
    target_logits_path = os.path.join(target_dir, "target_logits.npy")
    target_inmask_path = os.path.join(target_dir, "target_in_mask.npy")
    target_metadata_path = os.path.join(target_dir, "metadata.json")

    target_logits = np.load(target_logits_path)
    target_inmask = np.load(target_inmask_path)

    # Load metadata
    with open(target_metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"✅ Target model logits, inmask and metadata loaded from: {target_dir}")
    return target_logits, target_inmask, metadata

def loadShadowModelSignals(target_name: str, path: str = "processed_shadow_models"):
    """
    Loads logits and in_mask arrays for all shadow models inside:
        base_path / folder_name
    
    Returns:
        shadow_logits: ndarray of shape (N, M)
        shadow_inmasks: ndarray of shape (N, M)
    """
    shadow_dir = os.path.join(path, target_name)

    logits_list = []
    inmask_list = []

    index = 0
    while True:
        logits_path = os.path.join(shadow_dir, f"shadow_logits_{index}.npy")
        inmask_path = os.path.join(shadow_dir, f"in_mask_{index}.npy")

        # Stop when no more models exist
        if not os.path.exists(logits_path) or not os.path.exists(inmask_path):
            break

        logits_list.append(np.load(logits_path))
        inmask_list.append(np.load(inmask_path))

        index += 1

    # Stack into (N, M)
    shadow_logits_all = np.stack(logits_list, axis=1)
    shadow_inmask_all = np.stack(inmask_list, axis=1)

    print(f"✅ Loaded {index} shadow models from: {shadow_dir}")
    print(f"➡️ Logits shape: {shadow_logits_all.shape}")
    print(f"➡️ Inmask shape: {shadow_inmask_all.shape}")

    return shadow_logits_all, shadow_inmask_all

def loadFbdStudy(study_name: str, metadata: bool = True, gtl: bool = True, logits: bool = True):
    """
    Load FBD study output files from the study/<study_name>/trial_outputs directory.
    Files must follow the indexed naming scheme:
      metadata/metadata_{i}.json
      gtl_probabilities/gtl_probabilities_{i}.npy
      rescaled_logits/rescaled_logits_{i}.npy
    """
    study_dir = os.path.join("study", study_name)
    trial_outputs_dir = os.path.join(study_dir, "trial_outputs")

    meta_dir = os.path.join(trial_outputs_dir, "metadata")
    gtl_dir = os.path.join(trial_outputs_dir, "gtl_probabilities")
    logits_dir = os.path.join(trial_outputs_dir, "rescaled_logits")
    
    global_metadata_path = os.path.join(study_dir, "metadata.json")
    if os.path.isfile(global_metadata_path):
        with open(global_metadata_path, "r") as f:
            global_metadata = json.load(f)
    else:
        global_metadata = None

    fbd_trial_results = []
    gtl_list = []
    logits_list = []
    
    index = 0

    while True:
        loaded_any = False  # Detect if this index has any valid file
        # --- Metadata ---
        if metadata:
            meta_path = os.path.join(meta_dir, f"metadata_{index}.json")
            if os.path.isfile(meta_path):
                loaded_any = True
                with open(meta_path, "r") as f:
                    meta_dict = json.load(f)
                
                # Convert into dataclass
                fbd_trial_results.append(
                    FbdTrialResults(
                        accuracy     = meta_dict["accuracy"],
                        noise        = meta_dict["noise"],
                        centrality   = meta_dict["centrality"],
                        temperature  = meta_dict["temperature"],
                        tau          = meta_dict["tau"]
                    )
                )
            else:
                if index > 0:
                    break
        # --- GTL probabilities ---
        if gtl:
            gtl_path = os.path.join(gtl_dir, f"gtl_probabilities_{index}.npy")
            if os.path.isfile(gtl_path):
                loaded_any = True
                gtl_list.append(np.load(gtl_path))
            elif os.path.isdir(gtl_dir) and index == 0:
                pass
            elif os.path.isdir(gtl_dir):
                break
        # --- Rescaled logits ---
        if logits:
            logits_path = os.path.join(logits_dir, f"rescaled_logits_{index}.npy")
            if os.path.isfile(logits_path):
                loaded_any = True
                logits_list.append(np.load(logits_path))
            elif os.path.isdir(logits_dir) and index == 0:
                pass
            elif os.path.isdir(logits_dir):
                break

        if not loaded_any:
            break  # No files found for this index → finished

        index += 1

    print(f"✅ FbD study trial outputs loaded, amount: {index}")
    return global_metadata, fbd_trial_results, gtl_list, logits_list

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
