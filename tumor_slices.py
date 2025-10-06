import os
import numpy as np
from tqdm import tqdm

SOURCE_DIR = "preprocessed"              # your original data
TARGET_DIR = "tumor_slices"              # where tumor-only slices will go

os.makedirs(TARGET_DIR, exist_ok=True)

# Get all *_mask.npy files
mask_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith("_mask.npy")])

tumor_slice_count = 0
for mask_file in tqdm(mask_files, desc="Filtering tumor slices"):
    ct_file = mask_file.replace("_mask.npy", "_ct.npy")

    mask_path = os.path.join(SOURCE_DIR, mask_file)
    ct_path = os.path.join(SOURCE_DIR, ct_file)

    # Load the volumes
    mask_volume = np.load(mask_path)  # shape: (D, H, W)
    ct_volume = np.load(ct_path)      # shape: (D, H, W)

    base_name = mask_file.replace("_mask.npy", "")

    for i in range(mask_volume.shape[0]):
        if np.sum(mask_volume[i]) > 0:
            # Tumor slice found — save both CT and mask
            ct_slice = ct_volume[i]
            mask_slice = mask_volume[i]

            # Save slice
            ct_save_path = os.path.join(TARGET_DIR, f"{base_name}_slice_{i:03d}_ct.npy")
            mask_save_path = os.path.join(TARGET_DIR, f"{base_name}_slice_{i:03d}_mask.npy")

            np.save(ct_save_path, ct_slice)
            np.save(mask_save_path, mask_slice)

            tumor_slice_count += 1

print("\n✅ Tumor-only slices saved.")
print("🧬 Total tumor slices extracted:", tumor_slice_count)
print("📁 Saved to:", TARGET_DIR)
