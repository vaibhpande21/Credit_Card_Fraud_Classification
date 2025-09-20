# Credit Card Fraud Detection

This project builds a machine learning pipeline to detect fraudulent credit card transactions using the **Kaggle Credit Card Fraud Dataset (2013)**. The dataset is highly imbalanced, with fraud cases accounting for only **0.172%** of total transactions.

---

## 📌 Project Overview
- Dataset: **284,807 transactions** with **492 frauds**.
- Features: 
  - `Time` (seconds elapsed since first transaction)  
  - `Amount` (transaction value)  
  - `V1–V28` (PCA-transformed features)  
- Target: `Class` (0 = Non-Fraud, 1 = Fraud).
- Challenge: Detect rare fraud cases while minimizing **false negatives**.

---

## 🔍 Exploratory Data Analysis (EDA)
- No missing values in the dataset.
- Fraud transactions tend to have **higher average transaction values**.
- Fraud is **sparse across time**, scattered throughout the dataset.
- Features are **uncorrelated** due to PCA transformation.

---

## ⚙️ Methodology
1. **Preprocessing**
   - Standardization of features.
   - Addressed class imbalance using **SMOTE oversampling**.
2. **Models Implemented**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Neural Network (Keras Sequential)
3. **Evaluation Metrics**
   - Accuracy (misleading due to imbalance)  
   - Precision, Recall, F1-score  
   - ROC-AUC  
   - PR-AUC (preferred for imbalance)

---

## 📊 Results
- **Accuracy**: ~99% (misleading due to imbalance).  
- **Precision (Fraud class)**: ~15%  
  - Out of 100 predicted frauds, ~15 were actual frauds.  
- **Recall (Fraud class)**: ~87%  
  - Out of 100 actual frauds, ~87 were detected.  
- Strong recall indicates the model is effective at catching most fraud cases, but precision remains low — a tradeoff common in fraud detection.  

The **best-performing model** achieved:
- ROC-AUC: ~0.98  
- PR-AUC: ~0.85  

---

## 📈 Visualizations
- Fraud vs non-fraud transaction distribution.
- Correlation heatmap of PCA features.
- Transactions over time (fraud is sparse).
- Confusion matrices for models.

---

## 🚀 Tech Stack
- **Python**: pandas, numpy, scikit-learn, imbalanced-learn
- **XGBoost**
- **TensorFlow / Keras**
- **Matplotlib, Seaborn**

---

## 📦 Installation & Usage
```bash
# Clone repository
git clone https://github.com/vaibhpande21/Credit_Card_Fraud_Classification.git
cd Credit_Card_Fraud_Classification

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook fraud_detection.ipynb


