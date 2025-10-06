import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def overlay(ct, gt_mask, pred_mask, idx, output_path):
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(ct, cmap='gray')
    plt.title("CT Slice")

    plt.subplot(1,3,2)
    plt.imshow(ct, cmap='gray')
    plt.imshow(gt_mask, alpha=0.4, cmap='Reds')
    plt.title("Ground Truth")

    plt.subplot(1,3,3)
    plt.imshow(ct, cmap='gray')
    plt.imshow(pred_mask, alpha=0.4, cmap='Blues')
    plt.title("Predicted")

    plt.tight_layout()
    plt.savefig(f"{output_path}/overlay_{idx}.png")
    plt.close()

# 🔁 Load test data
X_test = np.load("tumor_slices/testing/ct_slices.npy")
y_test = np.load("tumor_slices/testing/mask_slices.npy")

# 🔁 Load trained model
model = tf.keras.models.load_model("unet_transfer_best_model.h5", compile=False)

# 🔁 Predict
y_pred = model.predict(X_test, batch_size=8)
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

# 🔁 Create output folder if it doesn't exist
output_path = "overlays"
os.makedirs(output_path, exist_ok=True)

# 🔁 Generate overlays for first 10 test slices
for idx in range(10):
    ct = X_test[idx].squeeze()
    gt = y_test[idx].squeeze()
    pred = y_pred_bin[idx].squeeze()

    overlay(ct, gt, pred, idx, output_path)
