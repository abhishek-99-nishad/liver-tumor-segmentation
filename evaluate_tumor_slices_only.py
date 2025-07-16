import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Persistent Counter File ---
COUNT_FILE = "slice_eval_count.txt"

def read_previous_count():
    if os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, "r") as f:
            try: 
                return int(f.read().strip())
            except:
                return 0
    return 0

def update_count(new_count):
    prev_count = read_previous_count()
    total_count = prev_count + new_count
    with open(COUNT_FILE, "w") as f:
        f.write(str(total_count))
    return total_count

# --- Load Model ---
model = load_model("unet_model.h5", compile=False)

# --- Load Preprocessed Data ---
DATA_DIR = "preprocessed"
ct_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])

# --- Evaluation Storage ---
y_trues = []
y_preds = []
slice_counter = 0  # For counting how many tumor slices evaluated in this run 

print("🔍 Evaluating slices that contain tumor pixels...")

for ct_file in tqdm(ct_files):
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct_path = os.path.join(DATA_DIR, ct_file)
    mask_path = os.path.join(DATA_DIR, mask_file)

    ct_volume = np.load(ct_path)
    mask_volume = np.load(mask_path)

    for i in range(ct_volume.shape[0]):
        ct_slice = ct_volume[i]
        mask_slice = mask_volume[i]

        if np.sum(mask_slice) == 0:
            continue  # Skip non-tumor slices

        slice_counter += 1  # ✅ Count this slice

        # Predict
        ct_input = np.expand_dims(ct_slice, axis=(0, -1))
        pred = model.predict(ct_input, verbose=0)[0, :, :, 0]
        pred_binary = (pred > 0.5).astype(np.uint8)

        y_trues.extend(mask_slice.flatten())
        y_preds.extend(pred_binary.flatten())

# --- Convert to numpy arrays
y_trues = (np.array(y_trues) > 0.5).astype(np.uint8)
y_preds = (np.array(y_preds) > 0.5).astype(np.uint8)

# --- Compute Metrics
acc = accuracy_score(y_trues, y_preds)
prec = precision_score(y_trues, y_preds, zero_division=0)
rec = recall_score(y_trues, y_preds, zero_division=0)
f1 = f1_score(y_trues, y_preds, zero_division=0)
miou = jaccard_score(y_trues, y_preds, zero_division=0)
intersection = np.sum(y_trues * y_preds)
dice = (2. * intersection) / (np.sum(y_trues) + np.sum(y_preds) + 1e-8)

# --- Update total slice count across runs
total_slices = update_count(slice_counter)

# --- Print Results
print("\n📊 Evaluation on Tumor-Only Slices:")
print(f"Evaluated slices this run : {slice_counter}")
print(f"Total slices evaluated (all runs): {total_slices}")
print(f"Accuracy        : {acc:.4f}")
print(f"Precision       : {prec:.4f}")
print(f"Recall          : {rec:.4f}")
print(f"F1 Score        : {f1:.4f}")
print(f"Dice Coefficient: {dice:.4f}")
print(f"Mean IoU        : {miou:.4f}")
