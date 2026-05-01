# MGMT-classification-using-DenseNet-and-clinical-fusion
# MGMT Methylation Prediction using 3D DenseNet

## 📌 Overview
This project focuses on predicting MGMT methylation status in glioblastoma patients using multi-modal 3D MRI data and deep learning.

A custom 3D DenseNet architecture is used to extract volumetric features from MRI scans, combined with clinical features such as age and gender using a gated fusion mechanism.

---

## 🧠 Key Features
- Multi-modal MRI input (T1, T1GD, T2, FLAIR)
- 3D DenseNet with Squeeze-and-Excitation (SE) attention
- Mixed global pooling (average + max)
- Clinical feature integration (age, gender)
- Gated fusion mechanism
- 5-Fold Stratified Cross Validation
- Temperature scaling for calibration
- Threshold optimization using OOF predictions
- Grad-CAM for model interpretability

---

## 📊 Results
- Mean CV AUC: **0.6382 ± 0.0937**
- OOF AUC: **0.6239**
- Accuracy: **63.36%**
- Macro F1-score: **0.6037**

---

## 📁 Project Structure
── BTP_final_notebook.ipynb # Main training and evaluation notebook
├── build_csv_upenn.py # Dataset CSV creation
├── preprocess_upenn.py # Preprocessing pipeline
├── preprocessing_verification_upenn.py # Preprocessing validation
