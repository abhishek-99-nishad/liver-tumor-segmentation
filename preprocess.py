import os
import nibabel as nib
import numpy as np
import cv2

def preprocess_and_save(ct_path, mask_path, save_dir="preprocessed", target_shape=(256, 256), normalize=True):
    """
    Preprocess a single CT + mask pair:
    - Resize to (256x256)
    - Normalize intensities
    - Save as .npy arrays
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load volumes
    ct = nib.load(ct_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    print(f"📦 Loaded {os.path.basename(ct_path)} — shape: {ct.shape}")

    pre_ct = []
    pre_mask = []

    for i in range(ct.shape[2]):
        ct_slice = ct[:, :, i]
        mask_slice = mask[:, :, i]

        # Resize using OpenCV
        ct_resized = cv2.resize(ct_slice, target_shape, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_slice, target_shape, interpolation=cv2.INTER_NEAREST)

        # Normalize CT intensities (optional)
        if normalize:
            ct_resized = (ct_resized - np.min(ct_resized)) / (np.max(ct_resized) - np.min(ct_resized) + 1e-5)

        pre_ct.append(ct_resized)
        pre_mask.append(mask_resized)

    # Convert to numpy arrays
    pre_ct = np.array(pre_ct)      # shape: [depth, 256, 256]
    pre_mask = np.array(pre_mask)

    # Save as .npy
    name = os.path.basename(ct_path).replace(".nii.gz", "").replace(".nii", "")
    np.save(os.path.join(save_dir, f"{name}_ct.npy"), pre_ct)
    np.save(os.path.join(save_dir, f"{name}_mask.npy"), pre_mask)

    print(f"✅ Saved: {name}_ct.npy and {name}_mask.npy → {save_dir}")


def batch_preprocess(ct_dir="ct_files", mask_dir="mask_files"):
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
    for ct_file in ct_files:
        ct_path = os.path.join(ct_dir, ct_file)
        mask_file = ct_file.replace("ct_", "mask_")
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"❌ Skipping {ct_file} — matching mask not found.")
            continue

        preprocess_and_save(ct_path, mask_path)
