# 📡 Telecom Customer Churn Prediction

A machine learning web app that predicts whether a telecom customer will churn or not, built with **Streamlit** and a **Stacking Classifier**.



---

## 📂 Project Structure

```
Telecom Customer Churn Project/
│
├── dataset/
│   └── Telco-Customer-Churn.csv
│
├── app.py               # Streamlit app
├── churn_notebook.ipynb # EDA + Model notebook
└── README.md
```

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Rows:** 7,043 customers
- **Target:** `Churn` — Yes / No
- **Features:** tenure, contract type, internet service, monthly charges, etc.

---

## 🤖 Model

A **Stacking Classifier** with 4 base models and a Logistic Regression meta-model:

| Base Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Non-linear patterns |
| Random Forest | Ensemble of trees |
| Gradient Boosting | Sequential boosting |

**Meta-model:** Logistic Regression with 5-fold cross-validation

### Results

| Metric | Score |
|---|---|
| ✅ Accuracy | ~81.7% |
| 📈 ROC-AUC | ~0.864 |

---


## 📋 App Pages

| Page | Description |
|---|---|
| EDA | Visualize churn distribution, contract types, and monthly charges |
| Train Model | Train the stacking model and view Accuracy + ROC-AUC |
| Predict | Enter customer details and get a churn prediction |

---

## 🛠️ Libraries Used

- `pandas` — data manipulation
- `scikit-learn` — machine learning models
- `matplotlib` / `seaborn` — data visualization
- `streamlit` — web app interface
