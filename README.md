Here’s a polished `README.md` for your project with clear sections and Markdown formatting:

````markdown
# Liver Disease Prediction

This project implements a **machine learning pipeline for liver disease prediction**. It was created as part of the BPC-UIM (Artificial Intelligence in Medicine) course at VUT Brno.

---

## Authors

- **Viktor Morovič** – 257026@vutbr.cz  
- **Filip Sedlár** – 262751@vutbr.cz  
- **Matúš Smolka** – 257044@vutbr.cz  

---

## Overview

The project consists of two main scripts:

1. **`main.py`** – Trains the predictive model using liver patient data and exports it as `model.pkl`.
2. **`testing.py`** – Loads a validation CSV dataset and the trained model (`model.pkl`) to predict disease labels and evaluate performance.

The goal is to classify patients as **Healthy** or **Patient** based on physiological biomarkers.

---

## Features

The model uses the following clinical features:

- Age
- Gender
- Total Bilirubin
- Direct Bilirubin
- Alkaline Phosphatase
- ALT (Alaninaminotransferase)
- AST (Aspartate Aminotransferase)
- Total Proteins
- Albumin
- Albumin/Globulin Ratio

Derived features include:

- AST/ALT Ratio
- Globulin (Total Protein - Albumin)
- Recalculated Albumin/Globulin Ratio

---

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Training the model (`main.py`)

```bash
python main.py
```

**Workflow in `main.py`:**

1. Load dataset from `liver-disease_data.csv`.
2. Preprocess data:

   * Encode gender and target labels
   * Handle missing and impossible values
3. Optional data visualization
4. Split data into train/test sets
5. Optimize hyperparameters using Optuna
6. Train final pipeline including:

   * Imputation
   * Scaling and power transform
   * SMOTE-Tomek for imbalance
   * XGBoost classifier
7. Evaluate via cross-validation and test set
8. Export trained pipeline as `model.pkl`

---

### 2. Testing / Validation (`testing.py`)

```bash
python testing.py
```

**Workflow in `testing.py`:**

1. Load an external CSV validation dataset.
2. Preprocess dataset using the same pipeline as `main.py`.
3. Load the trained model (`model.pkl`).
4. Predict disease labels.
5. Evaluate performance using **Matthews Correlation Coefficient (MCC)**.

---

## Evaluation Metrics

* **Matthews Correlation Coefficient (MCC)**
* Accuracy
* F1-score
* Cohen’s Kappa
* ROC-AUC
* Confusion Matrix visualization

The final model evaluation on the test set includes bootstrapped 95% confidence intervals for MCC.

---

## Visualizations

* Feature histograms and distributions
* Correlation matrix heatmaps
* Gender distribution plots
* Confusion matrices
* Feature importance plots

---

## File Structure

```
├── main.py           # Training script
├── testing.py        # Validation script
├── liver-disease_data.csv  # Training dataset
├── model.pkl         # Exported trained model
├── requirements.txt  # Dependencies
└── README.md
```

---

## Notes

* Pipelines ensure **data transformations happen within cross-validation folds**, avoiding data leakage.
* Imbalance in data is handled with **SMOTE-Tomek**.
* Hyperparameters are tuned using **Optuna** with MCC as the main metric.
* The scripts are designed for **robust clinical evaluation**, including bootstrapped confidence intervals.

---

## References

* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Scikit-Learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
* [Optuna Hyperparameter Optimization](https://optuna.org/)
* [Imbalanced-Learn Library](https://imbalanced-learn.org/)
