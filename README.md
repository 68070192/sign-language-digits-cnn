# Sign Language Digits Classification using Custom CNN

End-to-end Deep Learning project for classifying sign language hand gestures representing digits 0–9, developed and trained from scratch using PyTorch.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.55%25-brightgreen)
![Baseline Improvement](https://img.shields.io/badge/Over%20Baseline-%2B23.55%25-blueviolet)

---

## Project Overview

This project addresses the task of multiclass image classification on hand gesture digits (classes 0 to 9). The objective is to design, implement, and benchmark a Custom Convolutional Neural Network (CNN) against a standard linear baseline under strict reproducibility and leak-free experimental setups.

### Key Highlights
* Trained from Scratch: No pre-trained weights or transfer learning used; all convolutional layers and classification heads were trained from random initialization.
* Leakage-Free Pipeline: Strict split into Train (70%), Validation (15%), and Test (15%) sets, where data augmentations are exclusively applied to the training split.
* Strong Regularization: Combines Batch Normalization, Dropout (40%), Weight Decay, and Cosine Annealing Learning Rate scheduling.
* Significant Baseline Improvement: Custom CNN achieves 93.55% Test Accuracy, outperforming the single-layer baseline model (70.00%) by +23.55%.

---

## Dataset Information

* Dataset Name: Sign Language Digits Dataset
* Source: [Arda Mavi (Kaggle)](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset)
* Total Samples: 2,062 RGB images (resized to 64x64 pixels)
* Classes: 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
* Split Configuration (Seed: 42):
  * Train Set: 1,443 images (~70%)
  * Validation Set: 309 images (~15%)
  * Test Set: 310 images (~15%)

### Data Pipeline and Preprocessing
* Training Transforms: Resize (64x64) -> Random Rotation (+/- 12 degrees) -> Color Jitter (Brightness/Contrast 0.15) -> ToTensor -> Normalize (mean=0.5, std=0.5). (Note: Horizontal Flip was strictly omitted to preserve hand geometry and directional gestures).
* Validation / Test Transforms: Resize (64x64) -> ToTensor -> Normalize (mean=0.5, std=0.5).

---

## Model Architecture

### 1. Baseline Model (BaselineLinear)
A single-layer feed-forward network serving as a benchmark:
* Input: Flattened image vector (3 x 64 x 64 = 12,288 dimensions)
* Architecture: nn.Flatten() -> nn.Linear(12288, 10)
* Total Parameters: ~122,890

### 2. Proposed Custom CNN (SignLanguageCNN)
A 3-block convolutional neural network designed for spatial hierarchy extraction:


```text
Input Image [3, 64, 64]
│
├── Block 1: Conv2d(3 -> 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)   => [32, 32, 32]
├── Block 2: Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)  => [64, 16, 16]
├── Block 3: Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) => [128, 8, 8]
│
└── Classification Head:
    ├── AdaptiveAvgPool2d((1, 1))                                                    => [128, 1, 1]
    ├── Flatten()                                                                    => [128]
    ├── Dropout(p=0.4)
    └── Linear(128 -> 10)                                                            => [10] (Logits)
```
## Hyperparameters and Training Setup
| Parameter | Configuration |
| :--- | :--- |
| Framework | PyTorch 2.x (NVIDIA T4 GPU) |
| Loss Function | Cross-Entropy Loss (nn.CrossEntropyLoss) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | Cosine Annealing (T_max=45, eta_min=1e-5) |
| Batch Size | 32 |
| Epochs | 45 |
| Checkpointing | Model saved on lowest Validation Loss |
| Random Seed | 42 |

### Performance Comparison (Test Set)

| Model | Test Accuracy | Test Macro-F1 | Parameters |
| :--- | :---: | :---: | :---: |
| Baseline (Linear) | 70.00% | 0.7046 | 122.9K (122,890) |
| Custom CNN (Proposed) | 93.55% | 0.9358 | 95.0K (94,986) |
| Net Difference | +23.55% | +0.2312 | -27.9K (-27,904) |

> Observation: Despite having fewer trainable parameters than the dense baseline, the Custom CNN outperforms it by over 23%, verifying the necessity of translation invariance and spatial locality in computer vision tasks.

---

## Learning Curves and Confusion Matrix

<img width="2600" height="1000" alt="learning_curves" src="https://github.com/user-attachments/assets/a8547a5c-5a62-47fc-96f0-d7bbda34b64d" />


*Figure 1: Training and Validation Loss/Accuracy curves across 45 epochs showing steady convergence and controlled overfitting.*

<img width="1600" height="1600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/29aedb68-07d4-4946-90f9-b9fc184b10dd" />


*Figure 2: Confusion Matrix evaluated on the 310 unseen test samples.*

---

## Failure Case Analysis

Understanding model failure modes is crucial for identifying structural constraints:

<img width="2400" height="1200" alt="success_vs_failure" src="https://github.com/user-attachments/assets/59d05b6a-77c2-41bb-a8b5-417dc249d3e9" />


*Figure 3: Exemplary predictions on unseen test data comparing high-confidence success cases against failure modes.*

### Key Failure Modes

1. **Primary Confusion Pair (Class 3 Predicted as Class 8):**
   * Based on the test set Confusion Matrix, the most frequent misclassification occurs where true Class 3 is predicted as Class 8 (4 samples), while other errors—such as Class 6 and Class 7 predicted as Class 3 shown in the sampled failure figure above—occur only once each (1 sample).

2. **Ambiguity Under Low Confidence:**
   * Sampled misclassifications exhibit relatively low softmax probability scores (<50% confidence), indicating that errors primarily occur on visually ambiguous hand postures rather than high-confidence false predictions.

3. **Dataset Index Mapping Note:**
   * In the original dataset array mapping (`Y.npy`), certain class indices do not strictly align with the visual digit gesture (e.g., index 2 contains circular thumb-index gestures). While this index permutation does not affect multiclass classification metrics (Accuracy/Macro-F1), explicit label remapping is recommended for downstream real-time deployment.

---

## Project Structure

```text
sign-language-digits-cnn/
├── data/                       # (Ignored in Git) Raw dataset storage
├── notebooks/
│   └── experiment.ipynb        # Primary Google Colab execution notebook
├── outputs/                    # Exported evaluation figures
│   ├── confusion_matrix.png
│   ├── learning_curves.png
│   └── success_vs_failure.png
├── src/                        # Modular source code
├── .gitignore                  # Git tracking rules
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## How to Reproduce

### 1. Clone the Repository

```bash
git clone https://github.com/68070192/sign-language-digits-cnn.git
cd sign-language-digits-cnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run on Google Colab or Locally

* Open `notebooks/experiment.ipynb` in Google Colab (select **T4 GPU** runtime).
* Run all cells sequentially. The notebook will automatically download the dataset via `kagglehub`, format the directory structure, train the models, and save evaluation artifacts to `outputs/`.
