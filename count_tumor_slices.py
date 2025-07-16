# count_tumor_slices.py

import os
import numpy as np
from tqdm import tqdm

DATA_DIR = "preprocessed"
tumor_slice_count = 0
total_slices = 0

# Loop through all files that are *_mask.npy
all_mask_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

for mask_file in tqdm(all_mask_files, desc="Counting tumor slices"):
    mask_path = os.path.join(DATA_DIR, mask_file)
    mask = np.load(mask_path)

    total_slices += 1
    if np.sum(mask) > 0:
        tumor_slice_count += 1

print(f"\n🧮 Total Slices Evaluated     : {total_slices}")
print(f"🧬 Tumor-Containing Slices    : {tumor_slice_count}")
print(f"📊 Percentage Tumor Slices    : {(tumor_slice_count / total_slices) * 100:.2f}%")
