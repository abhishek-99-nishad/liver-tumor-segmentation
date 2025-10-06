import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load data
ct = np.load("preprocessed/P0006_ct_C2_ct.npy")           # shape: (depth, 256, 256)
true_mask = np.load("preprocessed/P0006_ct_C2_mask.npy")  # shape: (depth, 256, 256)

# Load trained model
model = load_model("unet_model.h5", compile=False)

# Prepare input
ct_input = np.expand_dims(ct, -1)  # shape: (depth, 256, 256, 1)

# Predict
pred_mask = model.predict(ct_input)
pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize prediction

# Choose slice index to visualize
slice_idx = 65  # You can change this to any valid index

# Plot side-by-side comparison
plt.figure(figsize=(15, 5))

# CT image
plt.subplot(1, 3, 1)
plt.imshow(ct[slice_idx], cmap="gray")
plt.title("CT Image")
plt.axis("off")

# Ground truth mask
plt.subplot(1, 3, 2)
plt.imshow(true_mask[slice_idx], cmap="Reds", alpha=0.8)
plt.title("Ground Truth Mask")
plt.axis("off")

# Predicted mask
plt.subplot(1, 3, 3)
plt.imshow(pred_mask[slice_idx, :, :, 0], cmap="Blues", alpha=0.8)
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()

# Save figure
output_path = f"comparison_slice_{slice_idx}.png"
plt.savefig(output_path)
print(f"✅ Saved comparison image to {output_path}")

plt.show()
