# Lung Disease Classification - Experimentation Repository

A comprehensive experimentation repository exploring various deep learning and traditional machine learning approaches for automated lung disease classification from chest X-ray images.

## Overview

This repository contains extensive experiments comparing traditional machine learning methods with state-of-the-art deep learning architectures for classifying chest X-ray images into three categories:
- **Normal** - Healthy lung conditions
- **Lung Opacity** - Various degrees of lung abnormalities  
- **Viral Pneumonia** - Viral pneumonia infections


## Dataset

**Source:** [Kaggle Lung Disease Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease/)

**Composition:**
- Total Images: 3,475 chest X-ray images
- Normal: 1,250 images
- Lung Opacity: 1,125 images
- Viral Pneumonia: 1,100 images


## Experimental Pipeline

### Traditional Machine Learning
- **Feature Extraction:** SIFT, HOG, LBP
- **Dimensionality Reduction:** PCA
- **Classifier:** Logistic Regression
- **Result:** 89.78% accuracy

### Deep Learning Models Tested

| Model | Accuracy (%) |
|-------|-------------|
| ResNet50 (Version 1) | 88.27 |
| ResNet50 (Version 2) | 89.17 |
| EfficientNetB0 | 88.87 |
| DenseNet | 85.11 |
| Custom CNN | 88.72 |
| Custom ANN | 88.72 |
| **Custom ANN (Revised)** | **91.2** |

### Optimization Techniques Applied
- **SMOTE** - Synthetic Minority Over-sampling for class balancing
- **Hyperparameter Tuning** - Learning rate, batch size, epochs optimization
- **Transfer Learning** - Fine-tuning pre-trained models
- **Explainability** - LIME and SHAP for model interpretability

## Best Model Performance

**Custom ANN (Revised)** achieved the highest accuracy:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.92 | 0.94 | 0.93 | 418 |
| Abnormal | 0.90 | 0.87 | 0.88 | 247 |
| **Overall Accuracy** | | | **0.91** | **665** |

- **Macro Average:** Precision 0.91, Recall 0.90, F1-Score 0.91
- **Weighted Average:** Precision 0.91, Recall 0.91, F1-Score 0.91

## Repository Contents

This repository contains multiple Jupyter notebooks exploring:

- Traditional ML approaches (SIFT + HOG + LBP)
- Deep learning architectures (ResNet, EfficientNet, DenseNet)
- Custom CNN and ANN implementations
- SMOTE data balancing experiments
- Model explainability visualizations (LIME, SHAP)
- Comparative analysis notebooks

## Technologies Used

- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV, scikit-image
- **Feature Extraction:** SIFT, HOG, LBP
- **Data Processing:** NumPy, Pandas, scikit-learn
- **Balancing:** SMOTE (imbalanced-learn)
- **Explainability:** LIME, SHAP
- **Visualization:** Matplotlib, Seaborn

## Key Findings

1. **Deep learning outperforms traditional ML** - Custom ANN (91.2%) vs Traditional ML (89.78%)
2. **SMOTE improves performance** - Balancing reduces bias toward majority class
3. **Hyperparameter tuning is crucial** - Revised ANN gained 2.5% accuracy improvement
4. **Explainability builds trust** - LIME/SHAP highlight critical diagnostic regions
5. **Transfer learning shows promise** - Pre-trained models provide competitive baselines

## Related

- **Deployment App:** [Lung Disease Classifier](https://lung-disease-classification.streamlit.app/)
- **Model Weights:** [HuggingFace](https://huggingface.co/lakshyalol/customann1)

---
