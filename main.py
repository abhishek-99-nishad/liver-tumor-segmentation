#part 1 : This part shows the full 3D tumor volume slice-by-slice and saves each one as an image. It's helpful to see how the tumor appears across the scan.”


# from visualize import show_and_save_volume

# ct_path = "ct_files/P0006_ct_C2.nii"        # 🔁 Use your actual filename
# mask_path = "mask_files/P0006_mask_C2.nii"

# show_and_save_volume(ct_path, mask_path)


#part 2 : This line preprocessed all scans and masks: resized them, normalized them, and saved them as .npy files — so the model can train faster and use less memory.”
from preprocess import batch_preprocess
batch_preprocess()


