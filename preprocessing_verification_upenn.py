import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DATASET_DIR = r"D:\processed_upenn"   # change if needed

patients = sorted(os.listdir(DATASET_DIR))

print(f"Total patients: {len(patients)}")

# ─── CHECK 1: FILE + SHAPE VALIDATION ─────────────────────────

bad_cases = []
shapes = []

for patient in tqdm(patients):
    p_path = os.path.join(DATASET_DIR, patient)

    try:
        vols = []
        for mod in ["T1", "T1GD", "T2", "FLAIR"]:
            f = os.path.join(p_path, f"{mod}.nii.gz")

            if not os.path.exists(f):
                raise Exception(f"Missing {mod}")

            img = nib.load(f)
            data = img.get_fdata()

            shapes.append(data.shape)
            vols.append(data)

        vols = np.stack(vols, axis=0)

        if vols.shape != (4, 96, 96, 96):
            bad_cases.append((patient, vols.shape))

    except Exception as e:
        bad_cases.append((patient, str(e)))

print("\n❌ Bad cases:", len(bad_cases))
print(bad_cases[:5])

# ─── CHECK 2: NORMALIZATION CHECK ─────────────────────────

means = []
stds = []

for patient in patients[:50]:  # sample first 50
    p_path = os.path.join(DATASET_DIR, patient)

    for mod in ["T1", "T1GD", "T2", "FLAIR"]:
        f = os.path.join(p_path, f"{mod}.nii.gz")
        data = nib.load(f).get_fdata()

        mask = data != 0
        if np.sum(mask) == 0:
            continue

        means.append(data[mask].mean())
        stds.append(data[mask].std())

print("\n📊 Normalization stats (foreground only):")
print(f"Mean (avg): {np.mean(means):.4f}")
print(f"Std  (avg): {np.mean(stds):.4f}")

# Expected:
# mean ≈ 0
# std ≈ 1

# ─── CHECK 3: VISUALIZATION (MOST IMPORTANT) ─────────────────

def show_patient(patient):
    p_path = os.path.join(DATASET_DIR, patient)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    for i, mod in enumerate(["T1", "T1GD", "T2", "FLAIR"]):
        f = os.path.join(p_path, f"{mod}.nii.gz")
        data = nib.load(f).get_fdata()

        mid = data.shape[2] // 2  # axial slice

        axes[i].imshow(data[:, :, mid], cmap='gray')
        axes[i].set_title(mod)
        axes[i].axis('off')

    plt.suptitle(patient)
    plt.show()


# 🔥 VISUAL CHECK (run multiple times)
for p in patients[:3]:
    show_patient(p)