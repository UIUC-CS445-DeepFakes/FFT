# Deepfake Detection using FFT + Machine Learning

This project implements a deepfake image detection pipeline using frequency-domain analysis (FFT) and a logistic regression classifier. The goal is to distinguish real vs fake images based on differences in their frequency spectra. The pipeline extracts handcrafted FFT-based features from images and trains a Logistic Regression classifier for binary deepfake classification.

## Overview

The pipeline follows these steps:

1. Load real and fake images from datasets (Google Drive)
2. Preprocess images (resize, grayscale, normalize)
3. Convert images to frequency domain using FFT
4. Extract statistical frequency features
5. Train a logistic regression classifier
6. Evaluate performance and visualize results

## Preprocessing
All images are preprocessed before feature extraction.

Image preprocessing
Resize images to 224×224
Convert images to grayscale
Normalize pixel values to [0,1]

The dataset is split into train / val / test sets: 22,702 images with an approximate 70% training, 15% testing, and 15% validation.

## Model Design

The FFT pipeline converts images into frequency space using a 2D Fast Fourier Transform. Statistical frequency-domain features are extracted and used to train a Logistic Regression classifier.

| Component                | Detail                                  |
| ------------------------ | --------------------------------------- |
| Feature Extraction       | 2D Fast Fourier Transform (FFT)         |
| Frequency Representation | Magnitude Spectrum                      |
| Statistical Features     | Mean, Std, Median, Min, Max             |
| Frequency Features       | Low-frequency and High-frequency Energy |
| Classifier               | Logistic Regression                     |
| Solver                   | `lbfgs`                                 |
| Max Iterations           | `1000`                                  |
| Feature Scaling          | `StandardScaler` normalization          |
| Output                   | Binary prediction (`Real` vs `Fake`)    |

## FFT Transformation

Each image is transformed into frequency space using:

f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude = np.log(np.abs(fshift) + 1)

The shifted FFT magnitude spectrum allows the model to analyze image frequency distributions.

Low-frequency regions capture large-scale image structure while high-frequency regions capture fine texture details and synthetic artifacts commonly found in deepfakes.

## Feature Extraction
The following statistical frequency-domain features are extracted from each image:

Mean frequency magnitude
Standard deviation
Median magnitude
Minimum magnitude
Maximum magnitude
Low-frequency mean and standard deviation
High-frequency mean and standard deviation
Low-frequency energy
High-frequency energy

These handcrafted FFT features are stored as NumPy .npy files to reduce runtime and avoid recomputing FFT transforms during later notebook executions.

## Training and Datasets
Our datasets were preprocessed, zipped, and stored in Google Drive. See the [Data Extraction Repository](https://github.com/UIUC-CS445-DeepFakes/Data-extraction) for more details.
During runtime, these zips files are copied from [Google Drive](https://drive.google.com/drive/folders/1D9VHNO4FI2E1Mxd78-GUCj0alPPMAdY0?usp=sharing) and into the Colab enviroment. This allows us to decrease file reading time.

During runtime, these zip files are copied from Google Drive into the Colab environment. This significantly decreases file reading time compared to reading directly from Drive during training.

The FFT pipeline uses three datasets:

| Dataset           | Purpose                          |
| ----------------- | -------------------------------- |
| CelebDFV2         | Real vs Deepfake celebrity faces |
| UADFV             | Video-based deepfake faces       |
| Human vs AI Faces | Real vs AI-generated human faces |

The extracted datasets are merged into combined train / validation / test splits.

## Model Training 

The Logistic Regression classifier is trained on extracted FFT feature vectors rather than raw images.

Feature matrices are standardized using StandardScaler() before training to improve convergence and classification performance.

Training runtime is significantly lower than the CNN pipeline because the model trains on compact handcrafted feature vectors rather than high-dimensional image tensors.

The trained model is saved using:
joblib.dump(model, "fft_logistic_model.pkl")

FFT feature matrices are saved using:
np.save(...)

This allows rapid experimentation without recomputing FFT features.

## Evaluation and Visualization

The model is evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

For visualization:

classification_report generates performance reports
confusion_matrix and ConfusionMatrixDisplay generate confusion matrices
matplotlib visualizes FFT magnitude spectra and evaluation plots

The FFT spectrum visualizations help demonstrate how frequency distributions differ between real and fake images.

## Runtime

The FFT + Logistic Regression pipeline is computationally lightweight compared to CNN-based methods.

| Method                    | Approximate Runtime      |
| ------------------------- | ------------------------ |
| FFT + Logistic Regression | Seconds                  |
| CNN (VGG16)               | Tens of minutes to hours |

The FFT pipeline was executed in Google Colab using a T4 GPU environment.

## Developers
- **Daniella Urteaga**: developed FFT model training and logistic regression classifier
- **Rebecca Samuel**: developed image preprocessor and visualizations

## Sources

1. Project Motivation:
   - Shahzad, H. F., Rustam, F., Flores, E. S., Luís Vidal Mazón, J., de la Torre Diez, I., & Ashraf, I. (2022). A Review of Image Processing Techniques for Deepfakes. Sensors (Basel, Switzerland), 22(12), 4556. https://doi.org/10.3390/s22124556
2. FFT Reference:
   - https://numpy.org/doc/stable/reference/routines.fft.html
   - https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
3. Logistic Regression:
   - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
4. Confusion Matrix Visualization:
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
5. Datasets:
   - CelebDFV2: https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset
   - UADFV: https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset
   - Human Faces Dataset: https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset