# correct_count_tumor_slices.py

import os
import numpy as np
from tqdm import tqdm

DATA_DIR = "preprocessed"
tumor_slices = 0
total_slices = 0

# Loop through all *_mask.npy files
all_mask_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

print("🔍 Scanning mask volumes...\n")

for mask_file in tqdm(all_mask_files, desc="Counting tumor-containing slices"):
    mask_path = os.path.join(DATA_DIR, mask_file)
    mask_volume = np.load(mask_path)  # shape: (depth, 256, 256)

    total_slices += mask_volume.shape[0]

    for slice_ in mask_volume:
        if np.any(slice_):  # Contains any non-zero value = tumor
            tumor_slices += 1

# Final stats
print("\n📊 Final Report")
print(f"🧠 Total Mask Volumes         : {len(all_mask_files)}")
print(f"📑 Total 2D Slices Scanned    : {total_slices}")
print(f"🧬 Tumor-Containing Slices    : {tumor_slices}")
print(f"📊 Percentage Tumor Slices    : {(tumor_slices / total_slices) * 100:.2f}%")


