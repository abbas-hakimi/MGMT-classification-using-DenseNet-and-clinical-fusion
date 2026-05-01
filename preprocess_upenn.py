import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────
DATASET_DIR  = r"D:\upenn gbm\PKG - UPENN-GBM-NIfTI\UPENN-GBM\NIfTI-files\images_structural"
OUTPUT_DIR   = r"D:\processed_upenn"
TARGET_SHAPE = (96, 96, 96)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── HELPERS ───────────────────────────────────────────────────

def load_nii(patient_folder, filename):
    filepath = os.path.join(patient_folder, filename + ".nii.gz")
    if not os.path.exists(filepath):
        return None, None
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


# ✅ Z-score normalization (foreground only)
def zscore_normalize(volume):
    mask = volume > 0
    vox = volume[mask]

    if len(vox) == 0:
        return volume.astype(np.float32)

    mean = vox.mean()
    std = vox.std()
    std = std if std > 1e-8 else 1.0

    volume = (volume - mean) / std
    volume[~mask] = 0

    return volume.astype(np.float32)


# ✅ Center crop or pad
def crop_or_pad(volume, target_shape):
    result = np.zeros(target_shape, dtype=np.float32)

    src_slices = []
    dst_slices = []

    for dim in range(3):
        s = volume.shape[dim]
        t = target_shape[dim]

        s0 = max(0, (s - t) // 2)
        d0 = max(0, (t - s) // 2)

        length = min(s - s0, t - d0)

        src_slices.append(slice(s0, s0 + length))
        dst_slices.append(slice(d0, d0 + length))

    result[tuple(dst_slices)] = volume[tuple(src_slices)]

    return result


# ─── MAIN ───────────────────────────────────────────────────────

MODALITIES = {
    "T1": "T1",
    "T1GD": "T1GD",
    "T2": "T2",
    "FLAIR": "FLAIR"
}

patients = sorted(os.listdir(DATASET_DIR))
print(f"Found {len(patients)} patients")

for patient in tqdm(patients):

    patient_path = os.path.join(DATASET_DIR, patient)
    if not os.path.isdir(patient_path):
        continue

    out_path = os.path.join(OUTPUT_DIR, patient)
    os.makedirs(out_path, exist_ok=True)

    for mod_key, mod_name in MODALITIES.items():

        filename = f"{patient}_{mod_name}"
        data, affine = load_nii(patient_path, filename)

        if data is None:
            print(f"Missing {mod_name} in {patient}")
            continue

        # ✅ FINAL CORRECT PIPELINE
        data = zscore_normalize(data)
        data = crop_or_pad(data, TARGET_SHAPE)

        # optional (stability)
        data = np.clip(data, -5, 5)

        # ✅ Preserve affine
        save_path = os.path.join(out_path, f"{mod_key}.nii.gz")
        nib.save(nib.Nifti1Image(data, affine), save_path)

print("✅ DONE")