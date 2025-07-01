import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def show_and_save_volume(ct_path, mask_path, save_dir="overlays"):
    """
    Display and save all slices from a 3D CT scan with tumor mask overlay.
    Each slice is saved in overlays/<scan_name>/slice_XXX.png.
    Each image is also shown for 1 second before auto-closing.
    """
    import time

    # Load volumes
    ct = nib.load(ct_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    num_slices = ct.shape[2]

    # Create subfolder: overlays/<scan_name>/
    scan_name = os.path.basename(ct_path).replace(".nii.gz", "").replace(".nii", "")
    folder_path = os.path.join(save_dir, scan_name)
    os.makedirs(folder_path, exist_ok=True)

    print(f"🧠 Visualizing & saving: {scan_name} ({num_slices} slices)")

    for i in range(num_slices):
        # Prepare figure
        plt.figure(figsize=(10, 5))
        plt.imshow(ct[:, :, i], cmap='gray')
        plt.imshow(mask[:, :, i], cmap='Reds', alpha=0.4)
        plt.title(f"{scan_name} - Slice {i+1}/{num_slices}")
        plt.axis('off')
        plt.tight_layout()

        # Save slice image
        slice_path = os.path.join(folder_path, f"slice_{i:03d}.png")
        plt.savefig(slice_path)

        # Log and show
        print(f"👁️ Showing slice {i+1}/{num_slices} — {slice_path}")
        plt.show(block=False)      # Show without waiting for user
        plt.pause(5.0)             # ⏱️ Show for 1 second
        plt.close()                # Auto-close window
