import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load preprocessed CT and ground truth mask
ct = np.load("preprocessed/P0006_ct_C2_ct.npy")
true_mask = np.load("preprocessed/P0006_ct_C2_mask.npy")

# Load the model
model = load_model("refinenet_model.h5", compile=False)

# Add channel dimension
ct = np.expand_dims(ct, axis=-1)  # Shape: (depth, 256, 256, 1)

# Predict
pred_mask = model.predict(ct)
pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize

# Dice function
def dice_score(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    intersection = np.sum(pred * true)
    return (2. * intersection) / (np.sum(pred) + np.sum(true) + 1e-5)

# Compute Dice score
dice = dice_score(pred_mask, true_mask)
print(f"🧪 Dice Similarity Coefficient: {dice:.4f}")
