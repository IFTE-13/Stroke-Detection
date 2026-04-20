# Brain Stroke Detection using Deep Learning
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

<p align="justify">This project focuses on evaluating multiple deep learning models for accurate brain stroke detection using neuroimaging data. It includes preprocessing techniques, dataset preparation, and model training using architectures such as VGG16, ResNet, DenseNet-201, and EfficientNet. A consistent experimental setup is applied to ensure fair comparison based on performance metrics like accuracy, precision, recall, and F1-score. The objective is to identify the most reliable and computationally efficient model for automated stroke diagnosis. It is designed to support research in medical imaging and assist in developing practical, data-driven clinical decision support systems.</p>

<p align="center">
  <img width="841" height="601" alt="Proposed Framework" src="https://github.com/user-attachments/assets/e21c0d38-fe03-4c47-b48c-6b90dab9f6e2" />
</p>

### 🎯 Key Objectives

- Compare performance of four state-of-the-art CNN architectures
- Implement transfer learning for medical image classification
- Provide explainable AI (XAI) visualizations for model decisions
- Identify the best-performing model for stroke detection

## Model Architectures

| Model | Parameters | Input Size | Key Feature |
|-------|------------|------------|--------------|
| **VGG16** | 138M | 224×224 | Simple, deep architecture |
| **ResNet50** | 25M | 224×224 | Residual connections |
| **DenseNet201** | 20M | 224×224 | Dense connectivity |
| **EfficientNetB0** | 5.3M | 224×224 | Efficient scaling |

## Dataset

<table align="center">
  <tr>
    <td align="center">
      <img width="350" height="350" alt="58 (1)" src="https://github.com/user-attachments/assets/c8fef44c-b798-4e73-a888-8b52c8a555d0" />
      <br />
      <sub><b>Figure 1:</b> Stroke </sub>
    </td>
    <td align="center">
      <img width="350" height="350" alt="49 (1)" src="https://github.com/user-attachments/assets/736e61ef-2e9e-4ef0-9115-770749be021d" />
      <br />
      <sub><b>Figure 2:</b> Normal </sub>
    </td>
  </tr>
</table>

> [!IMPORTANT]
> The complete project utilizes a custom dataset collected from two renowned hospitals, supplemented with images from additional online sources. Due to data privacy and institutional policies, the full hospital dataset cannot be publicly shared.

#### Dataset Structure:
```bash
dataset/
├── Training/ 
│       ├── Normal/ (# images)
│       └── Stroke/ (# images)
└─── Testing/
        ├── Normal/ (# images)
        └── Stroke/ (# images)
```

#### Demonstration Dataset
For model development, testing, and reproducibility, a publicly available brain CT dataset is used. This dataset is suitable for:

- Initial model architecture validation
- Pipeline testing and debugging  
- Code reproducibility across different environments

**Limitation:** This demonstration dataset is relatively small in size. Results obtained from this dataset may not fully represent real-world clinical performance. The custom hospital dataset provides more robust training and evaluation.

The demonstration dataset used in this repository is available at:
```bash
https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset
```

## Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
Jupyter Notebook
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/IFTE-13/Stroke-Detection.git
cd Stroke-Detection
```

### 2. Create virtual environmenty

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

> [!Tip]
> Recommended Execution Order
```bash
1. feature_extraction_comparison.ipynb  (Fast test)
   ↓
2. comprehensive_analysis.ipynb         (Full training)
   ↓
3. benchmark.ipynb                      (Final evaluation)
```

## Dependencies
```bash
tensorflow>=2.8.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
shap>=0.40.0
lime>=0.2.0
jupyter>=1.0.0
```

## Expected Outputs
1. Performance Metrics Table - Accuracy, Precision, Recall, F1, AUC
2. Confusion Matrices - For each model
3. ROC Curves - Comparative plot
4. Training History - Loss and accuracy curves
5. Grad-CAM Heatmaps - Model attention visualization
6. LIME Explanations - Feature importance per prediction
7. SHAP Values - Global feature importance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Author
MOHAMMED IFTEKHAR

GitHub: [@IFTE-13](https://github.com/IFTE-13)
