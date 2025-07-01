import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model
model = load_model("unet_model_best.h5")

# Choose a sample CT volume (preprocessed)
sample_ct = "preprocessed/P0006_ct_C2_ct.npy"
ct = np.load(sample_ct)  # Shape: [depth, 256, 256]

# Prepare for prediction
ct = np.expand_dims(ct, axis=-1)        # ➕ Add channel: [depth, 256, 256, 1]
#ct = np.transpose(ct, (0, 1, 2,))        # Just to ensure shape consistency

# Predict slice by slice
preds = []
for i in range(ct.shape[0]):
    slice_input = np.expand_dims(ct[i], axis=0)  # [1, 256, 256, 1]
    pred_mask = model.predict(slice_input)[0, :, :, 0]  # [256, 256]
    preds.append(pred_mask)

preds = np.array(preds)  # Shape: [depth, 256, 256]

# 🔍 Visualize a few slices
os.makedirs("predictions", exist_ok=True)
for i in range(0, len(preds), len(preds) // 5):  # Show 5 sample slices
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(ct[i, :, :, 0], cmap="gray")
    plt.title(f"CT Slice {i}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ct[i, :, :, 0], cmap="gray")
    plt.imshow(preds[i], cmap="Reds", alpha=0.4)
    plt.title(f"Predicted Tumor Mask")
    plt.axis('off')

    plt.tight_layout()
    save_path = f"predictions/slice_{i:03d}_overlay.png"
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved prediction: {save_path}")
