import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Define function to create overlay plots
def overlay(ct, gt_mask, pred_mask, idx, output_path, label):
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
    filename = f"{output_path}/{label}_vit_transfer_best_model_{idx}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved: {filename}")

# Load test data
X_test = np.load("tumor_slices/testing/ct_slices.npy")
y_test = np.load("tumor_slices/testing/mask_slices.npy")

# Load fine-tuned model
model = tf.keras.models.load_model("vit_transfer_best_model.h5", compile=False)

# Predict masks
y_pred = model.predict(X_test, batch_size=8)
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

# Fixed indices from transfer learning
top_5_idx = [130, 144, 202, 460, 595]
bottom_5_idx = [199, 545, 535, 378, 781]

# Create output directory
output_path = "vit_transfer_best_model"
os.makedirs(output_path, exist_ok=True)

# Generate overlays for top 5
print("\n🔝 Generating overlays for TOP 5 slices:")
for idx in top_5_idx:
    overlay(X_test[idx].squeeze(), y_test[idx].squeeze(), y_pred_bin[idx].squeeze(), idx, output_path, label="top")

# Generate overlays for bottom 5
print("\n🔻 Generating overlays for BOTTOM 5 slices:")
for idx in bottom_5_idx:
    overlay(X_test[idx].squeeze(), y_test[idx].squeeze(), y_pred_bin[idx].squeeze(), idx, output_path, label="bottom")
