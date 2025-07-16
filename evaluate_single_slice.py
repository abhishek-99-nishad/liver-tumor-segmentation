import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# --- Custom Losses (required if model uses bce + dice) ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# --- Load Model ---
model = load_model("unet_model.h5", custom_objects={"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss})
print("✅ Model loaded")

# --- Select file ---
DATA_DIR = "preprocessed"
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")]
all_files.sort()
print("\n✅ Available CT files:")
for i, f in enumerate(all_files):
    print(f"[{i}] {f}")

index = int(input("\n📌 Enter index of CT volume to evaluate (e.g. 0): "))
ct_file = all_files[index]
mask_file = ct_file.replace("_ct.npy", "_mask.npy")

# --- Load data ---
ct = np.load(os.path.join(DATA_DIR, ct_file))
mask = np.load(os.path.join(DATA_DIR, mask_file))

# --- Select slice ---
slice_idx = int(input(f"📌 Enter slice number (0 to {len(ct)-1}): "))
ct_slice = ct[slice_idx]
gt_mask = mask[slice_idx]

# --- Predict ---
input_tensor = np.expand_dims(ct_slice, axis=(0, -1))  # (1, 256, 256, 1)
pred_mask = model.predict(input_tensor)[0, ..., 0]  # (256, 256)
pred_binary = (pred_mask > 0.5).astype(np.uint8)

# --- Flatten and Binarize for Metrics ---
y_true = (gt_mask > 0.5).astype(np.uint8).flatten()
y_pred = pred_binary.flatten()

# --- Metrics ---
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
iou = jaccard_score(y_true, y_pred, zero_division=0)
dice = 2 * (np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

# --- Print results ---
print(f"\n📋 Metrics for {ct_file}, Slice {slice_idx}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"Dice:      {dice:.4f}")

# --- Visualization ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(ct_slice, cmap='gray')
plt.title("CT Slice")

plt.subplot(1, 3, 2)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(pred_binary, cmap='gray')
plt.title("Prediction")

plt.suptitle(f"{ct_file} - Slice {slice_idx}", fontsize=12)
plt.tight_layout()
filename = f"eval_{ct_file}_slice{slice_idx}.png"
plt.savefig(filename)
plt.show()
print(f"✅ Saved visualization as {filename}")
