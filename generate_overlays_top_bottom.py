import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import jaccard_score
import os

def overlay(ct, gt_mask, pred_mask, idx, output_dir, label):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(ct, cmap='gray')
    plt.title("CT Slice")

    plt.subplot(1, 3, 2)
    plt.imshow(ct, cmap='gray')
    plt.imshow(gt_mask, alpha=0.4, cmap='Reds')
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(ct, cmap='gray')
    plt.imshow(pred_mask, alpha=0.4, cmap='Blues')
    plt.title("Predicted")

    plt.tight_layout()
    filename = f"{output_dir}/{label}_overlayFineTune_{idx}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved: {filename}")

# Load test data
X_test = np.load("tumor_slices/testing/ct_slices.npy")
y_test = np.load("tumor_slices/testing/mask_slices.npy")

# Load trained fine-tuned model
model = tf.keras.models.load_model("unet_transfer_best_model.h5", compile=False)

# Predict
y_pred = model.predict(X_test, batch_size=8)
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

# Compute mIoU for each slice
miou_scores = []
for i in range(len(X_test)):
    y_true_flat = (y_test[i] > 0.5).astype(np.uint8).flatten()
    y_pred_flat = y_pred_bin[i].flatten()
    score = jaccard_score(y_true_flat, y_pred_flat, zero_division=0)
    miou_scores.append(score)

# Get top 5 and bottom 5 indices
top_5_idx = np.argsort(miou_scores)[-5:][::-1]
bottom_5_idx = np.argsort(miou_scores)[:5]

# Create output directory
output_dir = "FineTune Overlays"
os.makedirs(output_dir, exist_ok=True)

# Generate overlays for top 5
print("\n🔝 Generating overlays for TOP 5 slices:")
for idx in top_5_idx:
    overlay(X_test[idx].squeeze(), y_test[idx].squeeze(), y_pred_bin[idx].squeeze(), idx, output_dir, label="top")

# Generate overlays for bottom 5
print("\n🔻 Generating overlays for BOTTOM 5 slices:")
for idx in bottom_5_idx:
    overlay(X_test[idx].squeeze(), y_test[idx].squeeze(), y_pred_bin[idx].squeeze(), idx, output_dir, label="bottom")
