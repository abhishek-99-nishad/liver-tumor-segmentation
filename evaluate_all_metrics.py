import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from tqdm import tqdm

# Load model
model = load_model("unet_model_best.h5")

# Directories
DATA_DIR = "preprocessed"
all_ct_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])

# Lists for aggregated metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []
conf_matrices = []

print("🔍 Starting evaluation across all patients...")

for ct_file in tqdm(all_ct_files, desc="Evaluating"):
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct_path = os.path.join(DATA_DIR, ct_file)
    mask_path = os.path.join(DATA_DIR, mask_file)

    if not os.path.exists(mask_path):
        print(f"⚠️ Skipping {ct_file} — mask not found.")
        continue

    # Load and preprocess
    ct = np.load(ct_path)
    mask = np.load(mask_path)
    ct = np.expand_dims(ct, axis=-1)  # (slices, 256, 256, 1)
    preds = model.predict(ct, verbose=0)
    preds = (preds > 0.5).astype(np.uint8).squeeze()
    mask = mask.astype(np.uint8)

    # Flatten
    y_true = mask.flatten()
    y_pred = preds.flatten()

    # Metrics per patient
    accuracies.append(accuracy_score(y_true, y_pred))
    precisions.append(precision_score(y_true, y_pred, zero_division=0))
    recalls.append(recall_score(y_true, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    conf_matrices.append(confusion_matrix(y_true, y_pred))

    # ROC-AUC (optional)
    try:
        roc = roc_auc_score(y_true, y_pred)
        roc_aucs.append(roc)
    except ValueError:
        pass  # skip if only one class present

# Mean metrics
print("\n📊 Final Evaluation Summary (Averaged Across Patients):")
print(f"Accuracy:  {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall:    {np.mean(recalls):.4f}")
print(f"F1 Score:  {np.mean(f1_scores):.4f}")
if roc_aucs:
    print(f"ROC-AUC:   {np.mean(roc_aucs):.4f}")
else:
    print("ROC-AUC:   Not computable on all patients")

# Plot confusion matrix (average)
total_cm = sum(conf_matrices)
plt.figure()
plt.imshow(total_cm, cmap='Blues')
plt.title("Confusion Matrix (Summed)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.savefig("confusion_matrix.png")
plt.close()
print("✅ Saved: confusion_matrix.png")
