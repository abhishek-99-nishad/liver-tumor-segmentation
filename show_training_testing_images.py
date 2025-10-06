import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load all preprocessed CT and mask slices
DATA_DIR = "preprocessed"
all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])

X = []
Y = []

for ct_file in all_files:
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct_path = os.path.join(DATA_DIR, ct_file)
    mask_path = os.path.join(DATA_DIR, mask_file)

    ct = np.load(ct_path)
    mask = np.load(mask_path)

    X.extend(ct)
    Y.extend(mask)

X = np.expand_dims(np.array(X), axis=-1)
Y = np.expand_dims(np.array(Y), axis=-1)

print(f"Total Images: {X.shape[0]}")

# Train/Val Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training Images: {X_train.shape[0]}")
print(f"Validation Images: {X_val.shape[0]}")

# Function to display and save samples
def show_images(X_data, Y_data, tag="train", count=3):
    os.makedirs("sample_images", exist_ok=True)
    for i in range(count):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(X_data[i].squeeze(), cmap="gray")
        plt.title(f"{tag.capitalize()} Image {i+1}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(X_data[i].squeeze(), cmap="gray")
        plt.imshow(Y_data[i].squeeze(), cmap="Reds", alpha=0.4)
        plt.title(f"{tag.capitalize()} Mask {i+1}")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"sample_images/{tag}_sample_{i+1}.png")
        plt.close()
        print(f"✅ Saved: sample_images/{tag}_sample_{i+1}.png")

# Show 3 training and 3 validation images
show_images(X_train, Y_train, tag="train")
show_images(X_val, Y_val, tag="val")
