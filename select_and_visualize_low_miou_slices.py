import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import cv2

# --- Set paths ---
PREPROCESSED_DIR = 'preprocessed'
OUTPUT_DIR = 'visual_miou_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load models ---
unet = load_model('unet_model.h5', compile=False)
segnet = load_model('segnet_model.h5', compile=False)
refinenet = load_model('refinenet_model.h5', compile=False)

# --- mIoU calculation ---
def compute_miou(y_true, y_pred):
    return jaccard_score(y_true.flatten(), y_pred.flatten(), zero_division=1)

# --- Get all preprocessed CT and mask file pairs ---
all_files = sorted([f for f in os.listdir(PREPROCESSED_DIR) if f.endswith('_ct.npy')])
slice_scores = []

# Loop through all preprocessed *_ct.npy files
for ct_file in tqdm(all_files):
    mask_file = ct_file.replace('_ct.npy', '_mask.npy')
    ct_path = os.path.join(PREPROCESSED_DIR, ct_file)
    mask_path = os.path.join(PREPROCESSED_DIR, mask_file)

    if not os.path.exists(mask_path):
        continue

    # Load full 3D volumes
    ct_volume = np.load(ct_path)         # shape: (slices, 256, 256)
    mask_volume = np.load(mask_path)     # shape: (slices, 256, 256)

    # Loop through each slice
    for slice_idx in range(ct_volume.shape[0]):
        ct = ct_volume[slice_idx]
        mask = mask_volume[slice_idx]

        ct_resized = ct.astype(np.float32)
        mask_resized = mask.astype(np.uint8)

        if np.sum(mask_resized) == 0:
            continue  # Skip non-tumor slices

        # Normalize input
        input_tensor = ct_resized / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=(0, -1))  # (1, 256, 256, 1)
        mask_bin = (mask_resized > 0).astype(np.uint8)

        # Predict from all 3 models
        pred_unet = (unet.predict(input_tensor)[0, :, :, 0] > 0.5).astype(np.uint8)
        pred_segnet = (segnet.predict(input_tensor)[0, :, :, 0] > 0.5).astype(np.uint8)
        pred_refinenet = (refinenet.predict(input_tensor)[0, :, :, 0] > 0.5).astype(np.uint8)

        # Compute mIoUs
        miou_u = compute_miou(mask_bin, pred_unet)
        miou_s = compute_miou(mask_bin, pred_segnet)
        miou_r = compute_miou(mask_bin, pred_refinenet)

        avg_miou = np.mean([miou_u, miou_s, miou_r])
        max_dev = max(abs(miou_u - avg_miou), abs(miou_s - avg_miou), abs(miou_r - avg_miou))

        # ✅ Accept slices where all models have low mIoU (0.4 to 0.5) and predictions are similar (±0.05)
        if (
            0.4 <= miou_u <= 0.5 and
            0.4 <= miou_s <= 0.5 and
            0.4 <= miou_r <= 0.5 and
            max_dev <= 0.05
        ):
            slice_scores.append((
                f"{ct_file}_slice{slice_idx}",
                miou_u, miou_s, miou_r,
                ct_resized, mask_bin, pred_unet, pred_segnet, pred_refinenet
            ))

# Sort by average mIoU and pick top 5 low-mIoU consistent slices
slice_scores = sorted(slice_scores, key=lambda x: -np.mean(x[1:4]))[:5]

# --- Visualization ---
for i, (fname, m_u, m_s, m_r, ct_img, mask, pu, ps, pr) in enumerate(slice_scores):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{fname}\n mIoU - U-Net: {m_u:.4f}, SegNet: {m_s:.4f}, RefineNet: {m_r:.4f}", fontsize=14)

    axes[0, 0].imshow(ct_img, cmap='gray')
    axes[0, 0].set_title('Original CT')
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 2].imshow(pu, cmap='gray')
    axes[0, 2].set_title('U-Net Prediction')
    axes[0, 3].imshow(ps, cmap='gray')
    axes[0, 3].set_title('SegNet Prediction')

    axes[1, 0].imshow(pr, cmap='gray')
    axes[1, 0].set_title('RefineNet Prediction')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_{i+1}_{fname.replace('.npy', '')}.png"))
    plt.close()

print(f"✅ Done. Saved {len(slice_scores)} low-mIoU comparisons to: {OUTPUT_DIR}")
