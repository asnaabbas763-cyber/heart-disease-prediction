# heart-disease-prediction
Predicting heart disease using Cleveland Dataset — KNN, Logistic Regression &amp; Gradient Boosting
# 🫀 Heart Disease Prediction Using Patient Health Data

> A full Machine Learning lifecycle project — from raw clinical data to a deployed Streamlit application — for predicting cardiac risk using the Cleveland Heart Disease Dataset.

---

## 👥 Team Members & Contributions

| Stage | Stages Covered | Contributor |
|-------|---------------|-------------|
| Stages 1–4 | Problem Definition, Data Collection, Preprocessing & EDA | **Anagha** |
| Stages 5–8 | Feature Engineering, Model Building, Evaluation & Explainability | **Hiba Fathima** |
| Stages 9–10 | Deployment & Documentation | **Asna Abbas** |

**Course:** Predictive Analytics | Academic Year 2025–26

---

## 📌 Problem Statement & Motivation

Cardiovascular disease is one of the leading causes of death globally, yet early detection dramatically improves patient outcomes. Traditional clinical diagnosis often relies on multiple tests and expert judgement, which may not be accessible in resource-limited settings.

This project builds a machine learning classifier that predicts whether a patient has heart disease based on 13 clinical attributes — enabling fast, data-driven preliminary screening. The goal is to demonstrate how a full ML pipeline, from raw data to deployed web app, can support clinical decision-making.

---

## 📂 Dataset Description

| Property | Details |
|----------|---------|
| **Source** | [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) via Kaggle |
| **File** | `heart-disease.csv` |
| **Rows** | 303 patient records |
| **Features** | 13 clinical attributes + 1 target variable |
| **Class Distribution** | Heart Disease: **165 (54.5%)** · No Heart Disease: **138 (45.5%)** |

### Feature Glossary

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
| `thal` | Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect) | Categorical |
| `target` | Heart disease diagnosis (1 = disease, 0 = no disease) | **Target** |

---

## 🔬 Methodology — Full ML Life Cycle

### Stage 1 — Problem Definition *(Anagha)*
Framed the problem as a binary classification task. Defined success metrics (accuracy, precision, recall, F1-score), identified the clinical domain context, and reviewed existing literature on ML-based cardiac risk prediction.

### Stage 2 — Data Collection & Understanding *(Anagha)*
Sourced the Cleveland Heart Disease Dataset (303 records, 14 columns). Conducted initial profiling: data types, null counts, statistical summaries, and preliminary class balance checks.

### Stage 3 — Data Preprocessing & Cleaning *(Anagha)*
- Verified and handled missing or erroneous values
- Standardised numerical features using `StandardScaler`
- Encoded categorical variables appropriately
- Performed train/test split (80/20 stratified)
- Persisted processed splits as `data/X_train.pkl`, `data/X_test.pkl`, `data/y_train.pkl`, `data/y_test.pkl`

### Stage 4 — Exploratory Data Analysis *(Anagha)*
Conducted in `EDA.ipynb` and `EDA_cleaned.ipynb`:
- Age distribution analysis
- Target class distribution
- Correlation heatmap for all 13 features
- Boxplots to identify outliers across numerical variables
- Gender and chest-pain-type breakdowns

### Stage 5 — Feature Engineering & Selection *(Hiba Fathima)*
Applied two statistical selection methods to identify the most clinically significant features:
- **Mutual Information** scores (captures non-linear dependencies)
- **Chi-Square** tests (assesses categorical–target associations)

**Top 4 selected features:**
1. `cp` — Chest Pain Type
2. `thalach` — Maximum Heart Rate Achieved
3. `ca` — Number of Major Vessels
4. `thal` — Thalassemia / Blood Flow Classification

### Stage 6 — Model Building & Training *(Hiba Fathima)*
Three models were trained and compared using the cleaned, scaled dataset:
- **K-Nearest Neighbors (KNN)** — tuned with optimal `k`
- **Logistic Regression** — with L2 regularisation
- **Gradient Boosting** — ensemble tree-based method

### Stage 7 — Model Evaluation & Comparison *(Hiba Fathima)*

| Model | Accuracy | Notes |
|-------|----------|-------|
| 🏆 **KNN** | **90.16%** | Selected as production model |
| Logistic Regression | 85.25% | Strong linear baseline |
| Gradient Boosting | 77.05% | Performed below expectations on this dataset size |

Evaluation used accuracy, precision, recall, F1-score, and confusion matrix. Decision boundary plots were generated for KNN and Logistic Regression (see `plots/`).

### Stage 8 — Model Interpretation & Explainability *(Hiba Fathima)*
- Visualised decision boundaries for KNN and Logistic Regression classifiers
- Feature importance interpreted through Mutual Information (`plots/mutual_information.png`) and Chi-Square scores (`plots/chi_square.png`)
- Findings documented to ensure clinical transparency of predictions

### Stage 9 — Deployment *(Asna Abbas)*
The final KNN model (`heart_model.pkl`) and scaler (`scaler.pkl`) were serialised with `joblib` and deployed via **Streamlit**:
- Interactive sidebar for entering all 13 patient parameters
- Real-time prediction with risk classification
- Clinical cyberpunk UI theme with animated feedback

🔗 **Live App:** [https://nihalabiomed-heart-disease-prediction-app-vnrlqu.streamlit.app](https://nihalabiomed-heart-disease-prediction-app-vnrlqu.streamlit.app)

### Stage 10 — Documentation *(Asna Abbas)*
- Maintained structured Jupyter notebooks for all pipeline stages
- Wrote this README
- Organised the GitHub repository with branches: `main`, `feature/eda`, `feature/feature-selection`, `ml-models`

---

## 📊 Results Summary

| Metric | KNN (Production) | Logistic Regression | Gradient Boosting |
|--------|-----------------|--------------------|--------------------|
| Accuracy | **90.16%** | 85.25% | 77.05% |
| Selected Features | cp, thalach, ca, thal | cp, thalach, ca, thal | cp, thalach, ca, thal |
| Deployment | ✅ Live | — | — |

The KNN classifier achieved the highest accuracy and was selected for deployment. Feature selection via Mutual Information and Chi-Square tests reduced dimensionality from 13 to 4 most predictive clinical markers without sacrificing performance.

---

## 🖼️ Application Screenshots

Screenshots and EDA plots are available in the `plots/` directory. Key visuals include:

| Plot | File |
|------|------|
| Age Distribution | `plots/age_distribution.png` |
| Target Class Distribution | `plots/target_distribution.png` |
| Correlation Heatmap | `plots/correlation_heatmap.png` |
| Boxplots (outlier detection) | `plots/boxplots.png` |
| Mutual Information Scores | `plots/mutual_information.png` |
| Chi-Square Feature Scores | `plots/chi_square.png` |
| KNN Decision Boundary | `plots/decision_boundary_knn.png` |
| Logistic Regression Boundary | `plots/decision_boundary_lr.png` |

---

## 🛠️ Setup & Running Locally

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nihalabiomed/Heart-disease-prediction.git
cd Heart-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Dependencies (`requirements.txt`)
```
pandas
streamlit
joblib
scikit-learn
```

---

## 📁 Project Structure

```
Heart-disease-prediction/
│
├── app.py                              # Streamlit deployment app
├── heart-disease.csv                   # Raw Cleveland dataset (303 rows × 14 cols)
├── heart_model.pkl                     # Trained KNN model
├── scaler.pkl                          # Fitted StandardScaler
├── requirements.txt
│
├── data/
│   ├── X_train.pkl                     # Preprocessed training features
│   ├── X_test.pkl                      # Preprocessed test features
│   ├── y_train.pkl                     # Training labels
│   └── y_test.pkl                      # Test labels
│
├── EDA.ipynb                           # Exploratory Data Analysis (Anagha)
├── EDA_cleaned.ipynb                   # Cleaned EDA with final plots (Anagha)
├── feature_selection_and_model.ipynb   # Feature selection + model training (Hiba Fathima)
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

## 🔗 Links

- 🌐 **Live Deployment:** [Streamlit App](https://nihalabiomed-heart-disease-prediction-app-vnrlqu.streamlit.app)
- 📓 **Dataset Source:** [UCI ML Repository — Cleveland Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---

## 📄 Reference

> Heart Disease Prediction Using Patient Health Data — Full ML Life Cycle Project  
> Predictive Analytics Course | Academic Year 2025–26  
> Contributors: Anagha (Stages 1–4) · Hiba Fathima (Stages 5–8) · Asna Abbas (Stages 9–10)
