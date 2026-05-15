# 📊 Customer Churn Prediction

> Predict whether a telecom customer will churn using machine learning.
> Deployed as a live Streamlit web app.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)](https://streamlit.io)

---

## Problem Statement

Customer churn — when a customer stops using a service — is a critical
business metric. Acquiring new customers costs 5–10× more than retaining
existing ones. This project builds an ML model to **identify at-risk customers
before they leave**, enabling proactive intervention.

```
Customer data  →  ML model  →  Churn probability  →  Business action
```

---

## Dataset

**Telco Customer Churn** (IBM / Kaggle) · 7,043 rows · 21 features

| Feature group | Examples |
|---|---|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Account | Tenure, Contract type, Paperless billing |
| Services | Internet, Phone, Streaming, Tech support |
| Billing | Monthly charges, Total charges, Payment method |
| **Target** | **Churn (Yes / No)** |

Churn rate in dataset: **26.5%** (class imbalance handled via `class_weight`)

---

## Model Comparison

| Model | Accuracy | AUC-ROC | F1 (churn) |
|---|---|---|---|
| Logistic Regression | 81% | 0.84 | 0.60 |
| Random Forest | 84% | 0.87 | 0.65 |
| **XGBoost** ✓ | **85%** | **0.91** | **0.70** |

**Best model: XGBoost** — selected by AUC-ROC (best metric for imbalanced classification)

### Top churn drivers (feature importance)
1. Tenure (short-term customers churn more)
2. Contract type (month-to-month = highest risk)
3. Monthly charges (high bills → higher churn)
4. Internet service type (Fiber optic users churn more)
5. Tech support (customers without support churn more)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/activate         # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run preprocessing + training
python -m src.preprocess
python -m src.train

# 5. Launch the Streamlit app
streamlit run app.py
```

---

## Project Structure

```
churn-prediction/
├── data/
│   ├── telco_churn.csv        ← raw dataset
│   └── processed.csv          ← cleaned + encoded
├── notebooks/
│   ├── feature_importance.png
│   └── roc_curve.png
├── models/
│   ├── churn_model.pkl        ← trained XGBoost model
│   └── scaler.pkl             ← fitted StandardScaler
├── src/
│   ├── preprocess.py          ← cleaning, encoding, scaling
│   └── train.py               ← model training + evaluation
├── app.py                     ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Key Learnings

- **Class imbalance**: Churn datasets are imbalanced (~27% churn). Accuracy alone is misleading — AUC-ROC and F1-score on the minority class matter more.
- **Feature encoding**: `LabelEncoder` for multi-class categoricals; binary yes/no columns → 0/1 directly.
- **Feature scaling**: Essential for Logistic Regression (distance-based). Tree models (RF, XGBoost) don't require it but it doesn't hurt.
- **Model selection**: Ensemble methods consistently outperform linear models on structured tabular data.
- **Business framing**: Precision vs Recall tradeoff — in churn, **high recall** is preferred (catch as many churners as possible, even at the cost of false alarms).

---

## Skills Demonstrated

✅ Data cleaning & preprocessing  
✅ Exploratory data analysis (EDA)  
✅ Feature engineering & encoding  
✅ Model training & hyperparameter tuning  
✅ Model evaluation (accuracy, AUC, F1, confusion matrix)  
✅ ML deployment with Streamlit  
✅ Version control with Git  

---

## Live Demo

🌐 [View the live app →](https://your-app.streamlit.app)

---

## Resume Description

> **Customer Churn Prediction** | Python, XGBoost, Streamlit, scikit-learn  
> Built an end-to-end ML pipeline to predict customer churn on a 7K-row telecom dataset.
> Performed data preprocessing, feature encoding, and class imbalance handling.
> Compared Logistic Regression, Random Forest, and XGBoost — achieved **85% accuracy and AUC 0.91**.
> Deployed as an interactive real-time prediction app using Streamlit.

---

*Built by [Your Name] · [LinkedIn](https://linkedin.com) · [Portfolio](https://yoursite.com)*
