import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from tqdm import tqdm

# --- Custom Loss for loading model ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# --- Load model ---
model = load_model("refinenet_model.h5", custom_objects={"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss})

# --- Prepare data paths ---
DATA_DIR = "preprocessed"
all_ct_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])

# --- Metric accumulators ---
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []
conf_matrices = []
positive_class_accuracies = []  # new list

print("🔍 Starting evaluation across all patients...\n")

for ct_file in tqdm(all_ct_files, desc="Evaluating"):
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct_path = os.path.join(DATA_DIR, ct_file)
    mask_path = os.path.join(DATA_DIR, mask_file)

    if not os.path.exists(mask_path):
        print(f"⚠️ Skipping {ct_file} — mask not found.")
        continue

    # --- Load and predict ---
    ct = np.load(ct_path)
    mask = np.load(mask_path)
    ct = np.expand_dims(ct, axis=-1)  # (slices, 256, 256, 1)
    preds = model.predict(ct, verbose=0)
    preds = (preds > 0.5).astype(np.uint8).squeeze()
    mask = mask.astype(np.uint8)

    y_true = mask.flatten()
    y_pred = preds.flatten()

    # --- Metrics ---
    accuracies.append(accuracy_score(y_true, y_pred))
    precisions.append(precision_score(y_true, y_pred, zero_division=0))
    recalls.append(recall_score(y_true, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    conf_matrices.append(confusion_matrix(y_true, y_pred))

    # --- ROC-AUC ---
    try:
        roc = roc_auc_score(y_true, y_pred)
        roc_aucs.append(roc)
    except ValueError:
        pass

    # --- Positive class accuracy: TP / (TP + FP + FN) ---
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        pos_acc = tp / (tp + fp + fn + 1e-8)
        positive_class_accuracies.append(pos_acc)
    except:
        positive_class_accuracies.append(0.0)

# --- Mean metrics ---
print("\n📊 Final Evaluation Summary (Averaged Across Patients):")
print(f"Accuracy:                {np.mean(accuracies):.4f}")
print(f"Precision:               {np.mean(precisions):.4f}")
print(f"Recall:                  {np.mean(recalls):.4f}")
print(f"F1 Score:                {np.mean(f1_scores):.4f}")
print(f"Positive Class Accuracy: {np.mean(positive_class_accuracies):.4f}")
if roc_aucs:
    print(f"ROC-AUC:                 {np.mean(roc_aucs):.4f}")
else:
    print("ROC-AUC:                 Not computable")

# --- Confusion Matrix Plot ---
total_cm = sum(conf_matrices)
plt.figure()
plt.imshow(total_cm, cmap='Blues')
plt.title("Confusion Matrix (Summed)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.savefig("confusion_matrix_new.png")
plt.close()
print("✅ Saved: confusion_matrix_new.png")
