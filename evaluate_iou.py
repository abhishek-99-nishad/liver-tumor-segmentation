import os
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

# === Function to calculate IoU ===
def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

# === Load model ===
from tensorflow.keras.models import load_model

# Redefine loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# Load model with custom_objects
model = load_model("unet_model.h5", custom_objects={'bce_dice_loss': bce_dice_loss})

# print(f"📦 Loaded model: {model_path}")

# === Set data directory ===
DATA_DIR = "preprocessed"
all_ct_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])

ious = []

# === Loop through all CT volumes ===
for ct_file in tqdm(all_ct_files, desc="Evaluating IoU"):
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct_path = os.path.join(DATA_DIR, ct_file)
    mask_path = os.path.join(DATA_DIR, mask_file)

    if not os.path.exists(mask_path):
        print(f"❌ Skipping {ct_file}, mask not found.")
        continue

    ct = np.load(ct_path)        # shape: (depth, 256, 256)
    mask = np.load(mask_path)    # shape: (depth, 256, 256)

    ct = np.expand_dims(ct, axis=-1)  # (depth, 256, 256, 1)

    # Predict and threshold
    preds = model.predict(ct, verbose=0) > 0.5
    preds = preds.squeeze()
    mask = mask.astype(bool)

    # Compute IoU per slice
    for i in range(len(preds)):
        iou = compute_iou(mask[i], preds[i])
        ious.append(iou)

# === Final Mean IoU ===
mean_iou = np.mean(ious)
print(f"\n📏 Mean Intersection over Union (mIoU): {mean_iou:.4f}")
