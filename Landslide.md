![](https://i.postimg.cc/1XTPZcqk/Chat-GPT-Image-Aug-9-2026-01-25-51-PM.png)

# Hybrid CNN–Transformer + YOLOv12s Landslide Detection

##  Project Overview

The proposed system is a **hybrid deep-learning architecture** for landslide detection and segmentation from satellite, aerial, and drone imagery. Currently, the project is in the **training and development phase**. The main objective is to detect landslide regions and generate **pixel-level segmentation masks** for accurate localization.

![Screenshot](https://i.postimg.cc/QtbPJTGH/Screenshot-2026-08-11-163759.png)

## System Architecture

The overall pipeline is:

$$
\text{Input Image}
\rightarrow
\text{CNN Feature Extraction}
\rightarrow
\text{Transformer Attention}
\rightarrow
\text{Feature Projection}
\rightarrow
\text{Residual Fusion}
\rightarrow
\text{YOLOv12s Segmentation}
\rightarrow
\text{Detection + Mask Output}
$$

---

##  CNN Feature Extractor

The input RGB image passes through two convolutional layers:

$$
3 \rightarrow 32 \rightarrow 64
$$

Each convolutional layer uses **Batch Normalization** and **SiLU activation**.

The CNN extracts local features such as:

- Terrain edges
- Soil texture
- Vegetation patterns
- Geological boundaries

---

##  Transformer Attention

The resulting **64-channel feature map** is converted into a sequence and processed using **4-head Multi-Head Self-Attention (MHSA)**.

The Transformer captures **long-range spatial relationships** between different regions of the image. This helps the model understand broader terrain context rather than relying only on local patterns.

---

##  Residual Feature Enhancement

The Transformer output is projected from:

$$
64 \rightarrow 3
$$

using a **$1 \times 1$ convolution**.

The projected feature is then combined with the original RGB image using a residual connection:

$$
X_{\text{enhanced}} = X + F_{\text{projected}}
$$

The residual connection preserves the original image information while adding learned spatial and contextual features.

---

## YOLOv12s Segmentation

The enhanced representation is passed to **YOLOv12s Segmentation**.

The model generates:

- **Bounding boxes** — location of the detected landslide
- **Confidence scores** — prediction confidence
- **Segmentation masks** — exact affected region

Therefore, the system performs both **object detection** and **pixel-level segmentation**.

---

##  Training

The current training configuration uses:

- **Input resolution:** $640 \times 640$
- **Optimizer:** AdamW
- **Learning-rate scheduler:** Cosine learning-rate scheduling
- **Hardware:** GPU acceleration

Data augmentation includes:

- Mosaic
- MixUp
- Copy-Paste
- Rotation
- Scaling
- Horizontal/vertical flipping
- HSV augmentation

These augmentations improve model robustness against different **terrain, illumination, vegetation, and environmental conditions**.

---

##  Current Status

The landslide model is currently under **training and evaluation**. Once satisfactory performance is achieved, it will be integrated into the larger **real-time disaster monitoring system**.

In the final system, multiple satellite, aerial, or drone feeds can be analyzed simultaneously, and the detected disaster information can be passed to an **AI-based decision-making agent** for further monitoring and response.

---

##  Summary

| Component | Description |
|---|---|
| **Input** | Satellite, aerial, or drone RGB imagery |
| **CNN** | Extracts local spatial features |
| **Transformer** | Captures long-range spatial relationships |
| **Feature Projection** | Projects $64$ channels to $3$ channels |
| **Residual Fusion** | $X_{\text{enhanced}} = X + F_{\text{projected}}$ |
| **Detection Model** | YOLOv12s Segmentation |
| **Output** | Bounding boxes, confidence scores, and segmentation masks |
| **Resolution** | $640 \times 640$ |
| **Optimizer** | AdamW |
| **Scheduler** | Cosine Learning Rate |
| **Augmentation** | Mosaic, MixUp, Copy-Paste, Rotation, Scaling, Flipping, HSV |
| **Current Status** | Training and evaluation |
| **Future Integration** | Real-time disaster monitoring and AI decision-making agent |