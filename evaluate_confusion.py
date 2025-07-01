import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm

# Load model
model = load_model("unet_model_best.h5")  # or 'unet_model_best.h5' if using checkpoint

# Preprocessed data directory
DATA_DIR = "preprocessed"
all_ct_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])

# Initialize lists to collect metrics
all_precisions = []
all_recalls = []
all_f1s = []

for ct_file in tqdm(all_ct_files, desc="Evaluating"):
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct_path = os.path.join(DATA_DIR, ct_file)
    mask_path = os.path.join(DATA_DIR, mask_file)

    if not os.path.exists(mask_path):
        print(f"⚠️ Skipping {ct_file} — mask not found.")
        continue

    ct = np.load(ct_path)      # shape: (slices, 256, 256)
    mask = np.load(mask_path)  # shape: (slices, 256, 256)

    ct = np.expand_dims(ct, axis=-1)  # (slices, 256, 256, 1)
    preds = model.predict(ct, verbose=0)  # (slices, 256, 256, 1)
    preds = (preds > 0.5).astype(np.uint8).squeeze()  # threshold + remove last dim
    mask = mask.astype(np.uint8)

    # Flatten and compute metrics per volume
    y_true = mask.flatten()
    y_pred = preds.flatten()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)

# Final average metrics
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_f1 = np.mean(all_f1s)

print("\n📊 Evaluation Summary (Averaged Across Patients):")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall:    {avg_recall:.4f}")
print(f"F1 Score:  {avg_f1:.4f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming you collected all y_true and y_pred from each patient
all_y_true = []
all_y_pred = []

# Inside your loop (append like this)
y_true = mask.flatten()
y_pred = preds.flatten()
all_y_true.extend(y_true)
all_y_pred.extend(y_pred)

# After the loop
cm = confusion_matrix(all_y_true, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (All Patients)")
plt.savefig("confusion_matrix.png")
plt.show()
