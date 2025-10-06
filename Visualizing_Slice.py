import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Load test data
X_test = np.load("tumor_slices/testing/ct_slices.npy")
y_test = np.load("tumor_slices/testing/mask_slices.npy")

# Load fine-tuned model
model = tf.keras.models.load_model("unet_finetune_best_model.h5", compile=False)

# Choose the slice index you want to visualize
slice_index = 199

# Extract CT and mask slice
ct_slice = X_test[slice_index]
gt_mask = y_test[slice_index]

# Predict the mask using the model
pred_mask = model.predict(np.expand_dims(ct_slice, axis=0))[0]

# Threshold the prediction to get binary mask
pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_slice.squeeze(), cmap='gray')
axes[0].set_title(f"CT Slice #{slice_index}")
axes[0].axis("off")

axes[1].imshow(gt_mask.squeeze(), cmap='gray')
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")

axes[2].imshow(ct_slice.squeeze(), cmap='gray')
axes[2].imshow(pred_mask_bin.squeeze(), cmap='jet', alpha=0.5)
axes[2].set_title("Predicted Mask Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
