import os
import numpy as np
from tqdm import tqdm

def stack_slices(source_dir, save_path, suffix):
    all_slices = []

    for file in tqdm(sorted(os.listdir(source_dir)), desc=f"Stacking {os.path.basename(save_path)}"):
        if file.endswith(suffix):
            arr = np.load(os.path.join(source_dir, file))
            if arr.shape == (256, 256):
                all_slices.append(arr)
            else:
                print(f"⚠️ Skipping: {file} — shape: {arr.shape}")

    if not all_slices:
        print(f"❌ No valid slices found with suffix '{suffix}' in {source_dir}")
        return

    stacked = np.stack(all_slices)  # (N, 256, 256)
    stacked = np.expand_dims(stacked, axis=-1)  # (N, 256, 256, 1)
    np.save(save_path, stacked)
    print(f"✅ Saved {save_path} with shape: {stacked.shape}")

# === Run for all 4 files ===
stack_slices("tumor_slices/training", "tumor_slices/training/ct_slices.npy", "_ct.npy")
stack_slices("tumor_slices/training", "tumor_slices/training/mask_slices.npy", "_mask.npy")
stack_slices("tumor_slices/testing", "tumor_slices/testing/ct_slices.npy", "_ct.npy")
stack_slices("tumor_slices/testing", "tumor_slices/testing/mask_slices.npy", "_mask.npy")
