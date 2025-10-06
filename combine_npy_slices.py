import numpy as np
import os
import glob

def load_and_stack(folder, prefix):
    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_slice_*.npy")))
    print(f"🔍 Found {len(files)} files in '{folder}' with prefix '{prefix}'")

    if len(files) == 0:
        print("❌ No files found!")
        return np.array([])

    data = [np.load(f) for f in files]
    stacked = np.stack(data)
    print(f"✅ Stacked shape: {stacked.shape}")
    return stacked

# Create combined .npy files for training
X_train = load_and_stack("tumor_slices/training", "ct")
y_train = load_and_stack("tumor_slices/training", "mask")
np.save("tumor_slices/training/ct_slices.npy", X_train)
np.save("tumor_slices/training/mask_slices.npy", y_train)

# Create combined .npy files for testing
X_val = load_and_stack("tumor_slices/testing", "ct")
y_val = load_and_stack("tumor_slices/testing", "mask")
np.save("tumor_slices/testing/ct_slices.npy", X_val)
np.save("tumor_slices/testing/mask_slices.npy", y_val)

print("🎉 Combined and saved all .npy slices successfully!")
