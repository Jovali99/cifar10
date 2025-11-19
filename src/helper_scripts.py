import os
import numpy as np
import pickle

folder = "20251117_b2919d4746"
path = os.path.join("audit_signals", folder)

shadow_logits_path = os.path.join(path, "rescaled_shadow_model_logits.npy")
rescaled_shadow_model_logits = np.load(shadow_logits_path)
print(f"b4: {rescaled_shadow_model_logits.shape}")
rescaled_sm_logits = rescaled_shadow_model_logits.T
print(f"after: {rescaled_sm_logits.shape}")

np.save(shadow_logits_path, rescaled_sm_logits)

