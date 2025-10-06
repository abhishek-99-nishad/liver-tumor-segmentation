import os
import random
import shutil

# Configuration
SOURCE_DIR = "tumor_slices"
TRAIN_DIR = os.path.join(SOURCE_DIR, "training")
TEST_DIR = os.path.join(SOURCE_DIR, "testing")
SPLIT_RATIO = 0.8  # 80% train, 20% test

# Ensure output directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Step 1: Get all *_ct.npy files (use prefix for pairing with mask)
all_ct_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith('_ct.npy'))
slice_prefixes = [f.replace('_ct.npy', '') for f in all_ct_files]

# Step 2: Shuffle and split
random.seed(42)
random.shuffle(slice_prefixes)
split_index = int(len(slice_prefixes) * SPLIT_RATIO)
train_prefixes = slice_prefixes[:split_index]
test_prefixes = slice_prefixes[split_index:]

# Step 3: Copy paired slices
def copy_slices(prefixes, dest_dir):
    for prefix in prefixes:
        ct_path = os.path.join(SOURCE_DIR, f"{prefix}_ct.npy")
        mask_path = os.path.join(SOURCE_DIR, f"{prefix}_mask.npy")
        if os.path.exists(ct_path) and os.path.exists(mask_path):
            shutil.copy(ct_path, os.path.join(dest_dir, f"{prefix}_ct.npy"))
            shutil.copy(mask_path, os.path.join(dest_dir, f"{prefix}_mask.npy"))

copy_slices(train_prefixes, TRAIN_DIR)
copy_slices(test_prefixes, TEST_DIR)

print(f"✅ Tumor dataset split complete.")
print(f"📁 Training set: {len(train_prefixes)} slices")
print(f"📁 Testing set:  {len(test_prefixes)} slices")
