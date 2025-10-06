# evaluate_all_metrics.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, jaccard_score
import matplotlib.pyplot as plt

# Load data
X_test = np.load("tumor_slices/testing/ct_slices.npy")
y_test = np.load("tumor_slices/testing/mask_slices.npy")

# Load trained model
model = tf.keras.models.load_model("unet_finetune_final_model.h5", compile=False)

# Predict
y_pred = model.predict(X_test, batch_size=8)
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

# Flatten all for metrics
# Flatten all for metrics (force binary)
y_true_flat = (y_test.flatten() > 0.5).astype(np.uint8)
y_pred_flat = y_pred_bin.flatten()


# Metrics
acc = accuracy_score(y_true_flat, y_pred_flat)
prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
rec = recall_score(y_true_flat, y_pred_flat)
f1 = f1_score(y_true_flat, y_pred_flat)
dice = (2 * np.sum(y_true_flat * y_pred_flat)) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-7)
miou = jaccard_score(y_true_flat, y_pred_flat)
roc_auc = roc_auc_score(y_true_flat, y_pred.flatten())

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ Dice Coefficient: {dice:.4f}")
print(f"✅ Mean IoU (mIoU): {miou:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")

# ROC Curve (Optional)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_true_flat, y_pred.flatten())
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png")
plt.show()
