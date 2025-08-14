import os

PREPROCESSED_DIR = "preprocessed"
ct_files = sorted([f for f in os.listdir(PREPROCESSED_DIR) if f.endswith('_ct.npy')])

print(f"🧠 Total CT Volumes found: {len(ct_files)}\n")
for f in ct_files:
    print("•", f)
