<div align="center">

# 🫀 Heart Disease Prediction

### Predicting cardiac risk using the Cleveland Heart Disease Dataset
### K-Nearest Neighbors · Logistic Regression · Gradient Boosting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-prediction-3ycj5odppblq8pchewpxv4.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-90.16%25-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

</div>

---

## 📖 Overview

Cardiovascular disease is one of the leading causes of death globally, yet early detection dramatically improves patient outcomes. Traditional clinical diagnosis often relies on multiple tests and expert judgement — which may not be accessible in resource-limited settings.

This project builds a **binary ML classifier** that predicts whether a patient has heart disease based on **13 clinical attributes**, enabling fast, data-driven preliminary screening. It demonstrates a complete ML pipeline — from raw data all the way to a live deployed web application.

---

## 👥 Team Members & Contributions

| Contributor | Stages | Responsibilities |
|-------------|--------|-----------------|
| **Anagha Suresh** | Stages 1 – 4 | Problem Definition, Data Collection, Preprocessing & EDA |
| **Hiba Fathima M** | Stages 5 – 8 | Feature Engineering, Model Building, Evaluation & Explainability |
| **Asna Abbas** | Stages 9 – 10 | Deployment & Documentation |

> 📚 **Course:** Predictive Analytics &nbsp;|&nbsp; Academic Year 2025–26

---

## 🚀 Live Demo

> 🔗 **[heart-disease-prediction.streamlit.app](https://heart-disease-prediction-3ycj5odppblq8pchewpxv4.streamlit.app/)**

Enter patient parameters in the interactive sidebar and get a real-time cardiac risk prediction.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Source** | [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) via Kaggle |
| **File** | `heart-disease.csv` |
| **Records** | 303 patient records |
| **Features** | 13 clinical attributes + 1 target variable |
| **Class Split** | ❤️ Heart Disease: **165 (54.5%)** &nbsp;·&nbsp; ✅ No Disease: **138 (45.5%)** |

<details>
<summary><b>📋 Feature Glossary (click to expand)</b></summary>

<br>

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numerical |
| `sex` | Sex (1 = male, 0 = female) | Categorical |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | Categorical |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise-induced angina (1 = yes) | Categorical |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) | Numerical |
| `thal` | Thalassemia type (1 = normal, 2 = fixed, 3 = reversible) | Categorical |
| `target` | Heart disease diagnosis (1 = disease, 0 = none) | **Target** |

</details>

---

## 🔬 ML Pipeline — Full Life Cycle

### 🔹 Stage 1 — Problem Definition &nbsp;
Framed the problem as a **binary classification** task. Defined success metrics (accuracy, precision, recall, F1-score), identified the clinical domain context, and reviewed existing literature on ML-based cardiac risk prediction.

### 🔹 Stage 2 — Data Collection & Understanding &nbsp;
Sourced the Cleveland Heart Disease Dataset (303 records, 14 columns). Conducted initial profiling: data types, null counts, statistical summaries, and preliminary class balance checks.

### 🔹 Stage 3 — Data Preprocessing & Cleaning &nbsp;
- Verified and handled missing or erroneous values
- Standardised numerical features using `StandardScaler`
- Encoded categorical variables appropriately
- Performed **80/20 stratified train/test split**
- Persisted processed splits as `.pkl` files under `data/`

### 🔹 Stage 4 — Exploratory Data Analysis &nbsp;
Conducted across `EDA.ipynb` and `EDA_cleaned.ipynb`:
- Age and target class distribution
- Correlation heatmap across all 13 features
- Boxplots for outlier detection in numerical variables
- Gender and chest-pain-type breakdowns

### 🔹 Stage 5 — Feature Engineering & Selection &nbsp;
Applied two statistical methods to identify the most clinically significant features:

| Method | Purpose |
|--------|---------|
| **Mutual Information** | Captures non-linear feature–target dependencies |
| **Chi-Square Test** | Assesses categorical feature–target associations |

**Top 4 selected features:**

| # | Feature | Clinical Meaning |
|---|---------|-----------------|
| 1 | `cp` | Chest Pain Type |
| 2 | `thalach` | Maximum Heart Rate Achieved |
| 3 | `ca` | Number of Major Vessels (fluoroscopy) |
| 4 | `thal` | Thalassemia / Blood Flow Classification |

### 🔹 Stage 6 — Model Building & Training &nbsp;
Three models were trained and compared on the cleaned, scaled dataset:

| Model | Approach |
|-------|---------|
| **K-Nearest Neighbors (KNN)** | Tuned with optimal `k` via cross-validation |
| **Logistic Regression** | L2 regularisation |
| **Gradient Boosting** | Ensemble tree-based method |

### 🔹 Stage 7 — Model Evaluation & Comparison &nbsp;

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| 🏆 **KNN** | **90.16%** | — | — | — |
| Logistic Regression | 85.25% | — | — | — |
| Gradient Boosting | 77.05% | — | — | — |

> KNN was selected as the **production model** based on highest accuracy. Confusion matrices and decision boundary plots are available in `plots/`.

### 🔹 Stage 8 — Model Interpretation & Explainability &nbsp;
- Decision boundary plots generated for KNN and Logistic Regression
- Feature importance visualised via Mutual Information and Chi-Square scores
- Findings documented to ensure **clinical transparency** of predictions

### 🔹 Stage 9 — Deployment &nbsp;
The KNN model (`heart_model.pkl`) and scaler (`scaler.pkl`) were serialised with `joblib` and deployed via **Streamlit**:
- Interactive sidebar for entering all 13 patient parameters
- Real-time prediction with risk classification output
- Clinical cyberpunk UI theme with animated feedback

### 🔹 Stage 10 — Documentation &nbsp;
- Maintained structured Jupyter notebooks for all pipeline stages
- Authored this README and project PPT
- Organised GitHub repository with branches: `main`, `feature/eda`, `feature/feature-selection`, `ml-models`

---

## 📊 Results Summary

| Metric | 🏆 KNN (Production) | Logistic Regression | Gradient Boosting |
|--------|---------------------|--------------------|--------------------|
| **Accuracy** | **82%** | 80.33% | 81.97% |
| **Features Used** | cp, thalach, ca, thal | cp, thalach, ca, thal | cp, thalach, ca, thal |
| **Deployed** | ✅ Live | ❌ | ❌ |

Feature selection reduced dimensionality from **13 → 4** most predictive clinical markers without sacrificing performance.

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py                              # Streamlit deployment app
├── heart-disease.csv                   # Raw Cleveland dataset (303 × 14)
├── heart_model.pkl                     # Trained KNN model (joblib)
├── scaler.pkl                          # Fitted StandardScaler (joblib)
├── requirements.txt                    # Python dependencies
│
├── data/
│   ├── X_train.pkl                     # Preprocessed training features
│   ├── X_test.pkl                      # Preprocessed test features
│   ├── y_train.pkl                     # Training labels
│   └── y_test.pkl                      # Test labels
│
├── EDA.ipynb                           # EDA — Anagha Suresh
├── EDA_cleaned.ipynb                   # Cleaned EDA with plots — Anagha Suresh
├── feature_selection_and_model.ipynb   # Feature selection + training — Hiba Fathima M
│
└── plots/
    ├── age_distribution.png
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── boxplots.png
    ├── mutual_information.png
    ├── chi_square.png
    ├── decision_boundary_knn.png
    └── decision_boundary_lr.png
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 📦 Dependencies

```txt
streamlit
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
```

---

## 📜 License & Credits

This project is an academic coursework submission for the **Predictive Analytics** course (2025–26).  
Dataset: [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---

<div align="center">

Made with ❤️ by **Anagha Suresh**, **Hiba Fathima M**, and **Asna Abbas**

</div>
