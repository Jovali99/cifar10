import hashlib
import os
import numpy as np

def hashCfg(trainPath: str, audit: str = None, metadataPath: str, inmask: np.ndarray) -> str:
    # extracts the parameters of the train & audit configurations and hashes them along 
    # with the in_mask so each model will have a unique identifier
    hash = hashlib.sha256()
    # Hash train config
    hash.update()
    # Hash metadata
    hash.update()
    # Hash inmask
    hash.update()
    return hash

def save(savePath):
    if not os.path.exist(savePath):
        os.makedir(savePath)
        
    
