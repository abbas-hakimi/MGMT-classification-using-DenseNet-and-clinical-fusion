import pandas as pd
import os

# ── Load clinical CSV ──────────────────────────────────────────
df = pd.read_csv(r'D:\upenn gbm\UPENN-GBM_clinical_info_v2.1.csv')

# ── Keep baseline scans only (_11) ────────────────────────────
df = df[df['ID'].str.endswith('_11')].copy()

# ── Keep only Methylated / Unmethylated ───────────────────────
df = df[df['MGMT'].isin(['Methylated', 'Unmethylated'])].copy()

# ── Encode MGMT as binary label ───────────────────────────────
# Methylated   = 1
# Unmethylated = 0
df['MGMT_label'] = (df['MGMT'] == 'Methylated').astype(int)

# ── Encode Gender ─────────────────────────────────────────────
# M = 1, F = 0
df['Sex'] = (df['Gender'] == 'M').astype(int)

# ── Rename columns to match our pipeline ─────────────────────
df = df.rename(columns={'Age_at_scan_years': 'Age at MRI'})

# ── Add WHO CNS Grade (all GBM = grade 4) ────────────────────
df['WHO CNS Grade'] = 4

# ── Build folder path ─────────────────────────────────────────
PROCESSED_DIR = r'D:\processed_upenn'

df['folder_path'] = df['ID'].apply(
    lambda x: os.path.join(PROCESSED_DIR, x)
)

# ── Check which patients actually have processed files ────────
def all_files_exist(folder_path):
    required = ['T1.nii.gz', 'T1GD.nii.gz', 'T2.nii.gz', 'FLAIR.nii.gz']
    if not os.path.exists(folder_path):
        return False
    files = os.listdir(folder_path)
    return all(r in files for r in required)

df['has_files'] = df['folder_path'].apply(all_files_exist)

print(f"Total labeled patients  : {len(df)}")
print(f"With processed files    : {df['has_files'].sum()}")
print(f"Missing processed files : {(~df['has_files']).sum()}")

df = df[df['has_files']].reset_index(drop=True)

# ── Keep only needed columns ──────────────────────────────────
df_final = df[[
    'ID',
    'Sex',
    'Age at MRI',
    'WHO CNS Grade',
    'MGMT',
    'MGMT_label',
    'folder_path'
]].reset_index(drop=True)

print()
print(df_final.head())
print(f'\nShape: {df_final.shape}')
print()
print('MGMT distribution:')
print(df_final['MGMT'].value_counts())
print()
print('Label distribution:')
print(df_final['MGMT_label'].value_counts())

#── Save CSV ──────────────────────────────────────────────────
save_path = r'D:\processed_upenn\metadata_upenn.csv'
df_final.to_csv(save_path, index=False)
print(f'\nSaved to {save_path}')
# ```

# ---

# **Run it:**
# ```
# cd D:\mgmt
# python build_csv_upenn.py
# ```

# **Expected output:**
# ```
# Total labeled patients  : 262
# With processed files    : ~250-262
# Missing processed files : ~0-12

# Shape: (250+, 7)

# MGMT distribution:
# Unmethylated    151
# Methylated      111

# Label distribution:
# 0    151
# 1    111
# ```

# ---

# **After this runs successfully:**
# ```
# Step 1 ✅ Preprocessing done (D:\processed_upenn)
# Step 2 ✅ CSV built (metadata_upenn.csv)
# Step 3 → Upload to Kaggle (same way as UCSF)

# For uploading:
# kaggle datasets create -p D:\processed_upenn --dir-mode zip
# ```

# **One important note about file naming:**

# Your preprocessing script saves files as:
# ```
# T1.nii.gz
# T1GD.nii.gz    ← note: T1GD not T1c
# T2.nii.gz
# FLAIR.nii.gz