
#  Underwater Substrate Segmentation

## Overview

**SegSP** is a hybrid framework for underwater substrate segmentation using **deep learning semantic segmentation models** combined with **superpixel refinement (SNIC)**.

The framework is designed for **side-scan sonar imagery**. SegSP improves prediction quality by refining raw model outputs using superpixel majority voting.

### Key Features

- Supports multiple segmentation models
  - FCN8s
  - FCN32s
  - DeepLabV3
  - PSPNet
  - DenseASPP
- Prediction-only or evaluation mode
- Superpixel refinement using **SNIC**
- Inference pipeline for large sonar mosaics

---

# Pipeline

The SegSP inference pipeline:

Input Sonar Images  
↓  
Deep Learning Segmentation  
↓  
Raw Prediction  
↓  
Superpixel Refinement  
↓  
Final Segmentation Map

---

# Example Results
## Average Precision(%)

| Method | Raw Prediction | Superpixel refinement |
|------|------|------|
| Fcn8s | 60.6 | 63.73 |
| Fcn32s | 70.21 | 72.58 |
| Deeplabv3 | 75.15 | 78.34 |
| Pspnet | 77.77 | 80.30 |
| Denseaspp | 75.23 | 78.81 |

## F1-score of Shadow(%)
| Method | Raw Prediction | Superpixel refinement |
|------|------|------|
| Fcn8s | 47.7 | 50.07 |
| Fcn32s | 59.35 | 61.31 |
| Deeplabv3 | 59.91 | 63.82 |
| Pspnet | 60.98 | 63.88 |
| Denseaspp | 68.87 | 75.61 |


## Average F1-score(%)
| Method | Raw Prediction | Superpixel refinement |
|------|------|------|
| Fcn8s | 62.73 | 65.9 |
| Fcn32s | 71.1 | 72.75 |
| Deeplabv3 | 73.67 | 75.77 |
| Pspnet | 73.75 | 75.27 |
| Denseaspp | 74.84 | 77.72 |









---

# Installation

## System Requirements

- Python: **3.8 – 3.10**
- Recommended: **Python 3.9**
- GPU recommended
- Disk space ≥ **20 GB**

Check GPU:

```bash
nvidia-smi
```

---

# Install CUDA 11.8

Download from NVIDIA:

https://developer.nvidia.com/cuda-11-8-0-download-archive

Verify installation:

```bash
nvcc --version
```

Expected output:

```
Cuda compilation tools, release 11.8
```

---

# Create Conda Environment

```bash
conda create -n segsp python=3.9 -y
conda activate segsp
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```
True
```

---

# Install Dependencies

```bash
pip install ninja tqdm
pip install -r requirements.txt
```

---

# Install SNIC Superpixel Library

Navigate to:

```bash
cd SNIC-master/SNIC-master/snic_python_interface
```

Compile:

```bash
python compile_snic_lib.py
```

---

# Quick Start

## Run Inference + Evaluation

```bash
python runinference.py --model fcn8s
```

Supported models:

```
fcn8s
fcn32s
deeplabv3
psp
denseaspp
```

Example:

```bash
python runinference.py --model deeplabv3
```

---

## Prediction Only (No Evaluation)

```bash
python runinference.py --model deeplabv3 --pred_only --data_path (path to sonar images)
```

This will:

1. Generate raw predictions
2. Apply SNIC superpixel refinement
3. Save refined outputs

---

# Project Structure

```
SegSP
│
├── runinference.py
├── eval.py
│
├── core
│   ├── models
│   ├── data
│   └── utils
│
├── SNIC-master
│   └──SNIC-master
│      └──snic_python_interface
│          └──SNICdemo.py
│            
├── runs
│   └── raw predictions
│
└── superpixel_refinement
    └── refined predictions
```

---

# Citation

If you use this project in your research, please cite:

```
@article{SegSP2026,
  title={SegSP: Deep Learning and Superpixel Integration for Underwater Substrate Segmentation},
  author={Your Name},
  year={2026}
}
```

---

# Acknowledgements

This project is built on:

- SNIC Superpixel algorithm
- awesome-semantic-segmentation-pytorch

---

# Future Improvements

- Better boundary refinement
- Faster superpixel generation
- Support for larger sonar mosaics
