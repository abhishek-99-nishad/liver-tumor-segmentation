import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, jaccard_score, confusion_matrix, roc_curve
)
# Import all custom layers and functions from your model_vit.py file.
from model_vit import PatchExtractor, PatchEmbedding, combined_loss, dice_loss

# -----------------------------
# Load the full 827 test slices
# -----------------------------
# IMPORTANT: These file paths are now set to the correct files.
X_test = np.load("tumor_slices/testing/ct_slices.npy")[..., np.newaxis]
y_test = np.load("tumor_slices/testing/mask_slices.npy")[..., np.newaxis]

# -----------------------------
# Load the fine-tuned model with ALL custom objects
# -----------------------------
# It's crucial to pass all custom components that were used to build the model.
# The model name has been corrected to match the fine-tuning script's output.
custom_objects = {
    'PatchExtractor': PatchExtractor,
    'PatchEmbedding': PatchEmbedding,
    'combined_loss': combined_loss,
    'dice_loss': dice_loss,
}

model = tf.keras.models.load_model(
    "vit_transfer_best_model.keras",
    custom_objects=custom_objects,
    safe_mode=False
)

# -----------------------------
# Predictions
# -----------------------------
print("🔎 Running predictions...")
y_pred = model.predict(X_test, batch_size=16)

# Binarize predictions
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

# Flatten and ensure both arrays are the same data type
y_test_f = y_test.astype(np.uint8).flatten()
y_pred_f = y_pred_bin.flatten()

# -----------------------------
# Metrics
# -----------------------------
acc = accuracy_score(y_test_f, y_pred_f)
prec = precision_score(y_test_f, y_pred_f, zero_division=0)
rec = recall_score(y_test_f, y_pred_f, zero_division=0)
f1 = f1_score(y_test_f, y_pred_f, zero_division=0)

# Dice coefficient
dice = (2 * np.sum(y_test_f * y_pred_f)) / (
    np.sum(y_test_f) + np.sum(y_pred_f) + 1e-7
)

# Mean IoU
miou = jaccard_score(y_test_f, y_pred_f)

# ROC-AUC (use raw probabilities, not binary)
roc_auc = roc_auc_score(y_test_f, y_pred.flatten())

# -----------------------------
# Print metrics
# -----------------------------
print("\n📊 Evaluation Metrics:")
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ Dice Coefficient: {dice:.4f}")
print(f"✅ Mean IoU (mIoU): {miou:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test_f, y_pred.flatten())
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Transformer")
plt.legend()
plt.grid()
plt.savefig("roc_curve_transformer.png")
plt.close()

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test_f, y_pred_f)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Tumor", "Tumor"],
            yticklabels=["No Tumor", "Tumor"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Transformer")
plt.savefig("confusion_matrix_transformer.png")
plt.close()

print("\n📂 Saved plots: 'roc_curve_transformer.png', 'confusion_matrix_transformer.png'")
