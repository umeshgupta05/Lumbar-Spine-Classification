# 🧠 Lumbar Spine Degenerative Disease Classification

## 📌 Overview

This project aims to **automatically detect and classify lumbar spine degenerative diseases** from sagittal MRI images using deep learning techniques. The primary objective is to aid radiologists and clinicians in assessing common lumbar degenerative conditions, enabling faster diagnosis and improved patient outcomes.

### 💡 Objectives

- Detect presence of degeneration at intervertebral levels (L1/L2 to L5/S1)
- Classify severity (Mild, Moderate, Severe) of degenerative diseases
- Visualize per-disc predictions with heatmaps
- Enable reproducibility and analysis for clinical research and academic study

---

## 🧬 Dataset

We use a curated dataset of **lumbar spine MRI images** labeled by medical professionals. Each image is annotated with the presence and severity of the following conditions:

| Disease Name                     | Description                                      |
|----------------------------------|--------------------------------------------------|
| Left Neural Foraminal Narrowing | Nerve root compression on left side             |
| Right Neural Foraminal Narrowing| Nerve root compression on right side            |
| Left Subarticular Stenosis      | Narrowing of the spinal canal on left side      |
| Right Subarticular Stenosis     | Narrowing of the spinal canal on right side     |
| Central Spinal Canal Stenosis   | Narrowing in the center of spinal canal         |

### 🏷 Disc Levels

- L1/L2
- L2/L3
- L3/L4
- L4/L5
- L5/S1

> Labels are structured as a 5×5 matrix (5 diseases × 5 disc levels), each with severity grades: 0 = None, 1 = Mild, 2 = Moderate, 3 = Severe.

---

## 🧠 Model Architecture

We explore multiple deep learning approaches for both **detection** and **multi-label classification**:

- ✅ CNN Backbone: EfficientNet-B3 / ResNet-50
- 🔍 Preprocessing: CLAHE, normalization, spine cropping
- 🧱 Model Head: Multi-label sigmoid output for disease×disc prediction
- 🏷 Loss Function: BCEWithLogitsLoss + Weighted Focal Loss
- 📊 Evaluation: mAP, Accuracy, ROC-AUC, per-disease F1-score


