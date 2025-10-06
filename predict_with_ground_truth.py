import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("unet_model.h5")  # Use best model saved during training

# Load preprocessed CT volume and corresponding ground truth mask
ct_path = "preprocessed/P0006_ct_C2_ct.npy"
mask_path = "preprocessed/P0006_ct_C2_mask.npy"

ct = np.load(ct_path)   # shape: [depth, 256, 256]
mask = np.load(mask_path)  # shape: [depth, 256, 256]

# Prepare for prediction
ct = np.expand_dims(ct, axis=-1)  # ➕ Add channel: [depth, 256, 256, 1]

# Predict each slice
preds = []
for i in range(ct.shape[0]):
    slice_input = np.expand_dims(ct[i], axis=0)  # [1, 256, 256, 1]
    pred_mask = model.predict(slice_input, verbose=0)[0, :, :, 0]  # [256, 256]
    preds.append(pred_mask)

preds = np.array(preds)

# 📁 Create folder to save comparisons
os.makedirs("comparison", exist_ok=True)

# 🔍 Visualize 5 sample slices (e.g., slice 0, 21, 42, 63, 84)
for i in range(0, len(preds), len(preds) // 5):
    plt.figure(figsize=(15, 5))

    # Panel 1: Original CT slice
    plt.subplot(1, 3, 1)
    plt.imshow(ct[i, :, :, 0], cmap="gray")
    plt.title(f"CT Slice {i}")
    plt.axis('off')

    # Panel 2: Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(ct[i, :, :, 0], cmap="gray")
    plt.imshow(mask[i], cmap="Greens", alpha=0.4)
    plt.title("Ground Truth Tumor Mask")
    plt.axis('off')

    # Panel 3: Predicted tumor mask
    plt.subplot(1, 3, 3)
    plt.imshow(ct[i, :, :, 0], cmap="gray")
    plt.imshow(preds[i], cmap="Reds", alpha=0.4)
    plt.title("Predicted Tumor Mask")
    plt.axis('off')

    plt.tight_layout()
    save_path = f"comparison/slice_{i:03d}_compare.png"
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved: {save_path}")
