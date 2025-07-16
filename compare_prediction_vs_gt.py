import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Config ===
model_path = "unet_model.h5"  # or "unet_model_best.h5"
ct_path = "preprocessed/P0006_ct_C2_ct.npy"
mask_path = "preprocessed/P0006_ct_C2_mask.npy"
slice_index = 65
save_path = f"comparison_slice_{slice_index:03d}.png"

# === Load Model ===
model = load_model(model_path)
print(f"✅ Loaded model: {model_path}")

# === Load Data ===
ct = np.load(ct_path)
gt_mask = np.load(mask_path)
print(f"🧠 Loaded slice {slice_index} from: {ct_path}")

# === Prepare Prediction ===
ct_slice = ct[slice_index]
gt_slice = gt_mask[slice_index]

input_slice = np.expand_dims(ct_slice, axis=(0, -1))  # shape: (1, 256, 256, 1)
pred = model.predict(input_slice)[0, :, :, 0]          # shape: (256, 256)

# === Plot ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(ct_slice, cmap="gray")
plt.title("CT Slice")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ct_slice, cmap="gray")
plt.imshow(pred, cmap="Reds", alpha=0.4)
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ct_slice, cmap="gray")
plt.imshow(gt_slice, cmap="Blues", alpha=0.4)
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig(save_path)
plt.show()

print(f"✅ Saved comparison image as {save_path}")
