# Deepfake Detection using FFT + Machine Learning

This project implements a deepfake image detection pipeline using frequency-domain analysis (FFT) and a logistic regression classifier. The goal is to distinguish real vs fake images based on differences in their frequency spectra.

## Overview

The pipeline follows these steps:

1. Load real and fake images from datasets (Google Drive)
2. Preprocess images (resize, grayscale, normalize)
3. Convert images to frequency domain using FFT
4. Extract statistical frequency features
5. Train a logistic regression classifier
6. Evaluate performance and visualize results

## Developers
- **Daniella Urteaga**: developed FFT model training and logistic regression classifier
- **Rebecca Samuel**: developed image preprocessor and visualizations

## Datasets

We used three datasets found online: [CelebDF](https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset), [Human Faces](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset ) and [UADFV_dataset](https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset). We saved these datasets to Google Drive and used Google Colab to go through the deepfake detection pipeline.

## Development
Clone the repository in Google Colab:
```
!git clone https://github.com/UIUC-CS445-DeepFakes/FFT.git
```
In Google Colab, import the following:
<code>
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt</code>
<code>
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from glob import glob</code>

## Model Design

- Preprocessing
  - Images are resized to 224 x 224, converted to grayscale, and normalized
- FFT Transformation
  - Each image is converted into frequency space using 2D FFT
- Feature Extraction
  - We extract statistical features from the frequency domain such as mean frequency magnitude, standard deviation, low-frequency energy, and high-frequency energy.
- Model Training
  - We use a logistic regression classifier to train the model.
