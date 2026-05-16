import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard AI | Cardiac Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. GLOBAL CSS  (Warm Clinical Emerald Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1a14;
    color: #d4e8d8;
}
.main { background-color: #0d1a14; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1810 0%, #112318 100%);
    border-right: 1px solid rgba(52,211,153,0.15);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p { color: #a7c8b0 !important; }

.stButton > button {
    background: linear-gradient(135deg, #34d399, #059669);
    color: #0a1810;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    border: none;
    border-radius: 8px;
    height: 3.2em;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(52,211,153,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f97316, #dc2626);
    box-shadow: 0 0 24px rgba(249,115,22,0.4);
    transform: translateY(-2px);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f2018, #162c1e);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 12px;
    padding: 18px 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #34d399 !important;
}
[data-testid="stMetricLabel"] { color: #6b9e7a !important; }

h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #34d399 !important;
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(52,211,153,0.35);
}
h2, h3 { color: #a3dbb4 !important; font-family: 'DM Sans', sans-serif !important; }
hr { border-color: rgba(52,211,153,0.15) !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1a14; }
::-webkit-scrollbar-thumb { background: #1e4d33; border-radius: 4px; }

@keyframes pulse-ring {
    0%   { transform: scale(0.8); opacity: 1; }
    100% { transform: scale(2.4); opacity: 0; }
}
.pulse-dot {
    position: fixed;
    border-radius: 50%;
    background: transparent;
    border: 3px solid #34d399;
    animation: pulse-ring 1.8s ease-out infinite;
    z-index: 9999;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LOAD MODEL ASSETS
#    Stage 5 saved: models/scaler.pkl, models/top_features.pkl
#    Stage 7 saved: models/best_model.pkl  (best of KNN / LR / GB)
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model        = joblib.load('models/best_model.pkl')
    scaler       = joblib.load('models/scaler.pkl')
    top_features = joblib.load('models/top_features.pkl')
    return model, scaler, top_features

try:
    model, scaler, TOP_FEATURES = load_assets()
    model_loaded = True
    model_labels = {
        'KNeighborsClassifier'       : 'K-Nearest Neighbors (KNN)',
        'LogisticRegression'         : 'Logistic Regression',
        'GradientBoostingClassifier' : 'Gradient Boosting'
    }
    model_label = model_labels.get(type(model).__name__, type(model).__name__)
except FileNotFoundError as e:
    st.error(f"⚠️ Asset not found: {e}  |  Run Stage 5–7 notebooks to generate models/.")
    model_loaded = False
    TOP_FEATURES = ['cp', 'thalach', 'ca', 'thal', 'oldpeak', 'exang', 'slope', 'restecg']
    model_label  = 'N/A'

# ─────────────────────────────────────────────
# 4. SIDEBAR — PATIENT INPUTS
#    All 13 features collected; scaler applied to all 13; only Top-8 fed to model.
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 24px 0;'>
        <div style='font-size: 36px;'>🫀</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 18px;
                    color: #34d399; letter-spacing: 4px;'>HEARTGUARD</div>
        <div style='font-size: 10px; color: #2d6645; letter-spacing: 3px; margin-top:4px;'>
            PREDICTIVE RISK ENGINE v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**👤 Patient Demographics**")
    age     = st.number_input("Age (years)", min_value=1, max_value=110, value=52)
    sex     = st.radio("Biological Sex", ["Male", "Female"], horizontal=True)
    sex_val = 1 if sex == "Male" else 0

    st.markdown("---")
    st.markdown("**🩺 Clinical Measurements**")
    cp = st.selectbox("Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Typical Angina",
                1: "1 – Atypical Angina",
                2: "2 – Non-Anginal Pain",
                3: "3 – Asymptomatic"
            }[x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol     = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                    format_func=lambda x: "Yes (1)" if x else "No (0)")
    restecg  = st.selectbox("Resting ECG Result",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "0 – Normal",
                        1: "1 – ST-T Wave Abnormality",
                        2: "2 – LV Hypertrophy"
                    }[x])

    st.markdown("---")
    st.markdown("**📊 Stress Test Results**")
    thalach = st.slider("Maximum Heart Rate Achieved (bpm)", 60, 220, 150)
    exang   = st.radio("Exercise-Induced Angina", [0, 1], horizontal=True,
                    format_func=lambda x: "Yes (1)" if x else "No (0)")
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope   = st.selectbox("Slope of Peak Exercise ST",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "0 – Upsloping",
                        1: "1 – Flat",
                        2: "2 – Downsloping"
                    }[x])

    st.markdown("---")
    st.markdown("**🔬 Angiography & Thalassemia**")
    ca   = st.selectbox("Major Vessels Coloured (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 – Normal",
                    1: "1 – Fixed Defect",
                    2: "2 – Reversible Defect",
                    3: "3 – Unknown"
                }[x])

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:10px; color:#2d6645; text-align:center; line-height:2;'>
        ACTIVE FEATURES (Top-8)<br>
        <span style='color:#34d399;'>{' · '.join(TOP_FEATURES)}</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. HEADER & LIVE METRICS STRIP
# ─────────────────────────────────────────────
st.markdown("<h1>🫀 HEARTGUARD — CARDIAC RISK PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:#6b9e7a; font-size:13px; margin-top:-12px; letter-spacing:1px;'>"
    f"Cleveland Heart Disease Dataset · Best Model: {model_label} · "
    f"Feature Selection: Mutual Information + Chi-Square (Top-8 of 13)</p>",
    unsafe_allow_html=True
)
st.markdown("---")

target_hr  = 220 - age
hr_pct     = round((thalach / target_hr) * 100, 1) if target_hr else 0
chol_flag  = "⚠ HIGH" if chol > 240 else "✓ OK"
bp_flag    = "⚠ HIGH" if trestbps > 140 else "✓ OK"
exang_flag = "⚠ YES" if exang else "✓ NO"

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Target HR (Age-Max)",  f"{target_hr} bpm",  f"Achieved: {hr_pct}%")
c2.metric("🩸 Cholesterol",          f"{chol} mg/dL",      chol_flag)
c3.metric("💉 Resting BP",           f"{trestbps} mmHg",   bp_flag)
c4.metric("🏃 Exercise Angina",      "Present" if exang else "Absent", exang_flag)

st.markdown("---")

# ─────────────────────────────────────────────
# 6. MAIN PANEL — MODEL INFO + INFERENCE
# ─────────────────────────────────────────────
left, right = st.columns([3, 2])

with right:
    st.markdown("### ⚙️ Model Pipeline")
    st.markdown(f"""
    <div style='background:#0f2018; border:1px solid rgba(52,211,153,0.18);
                border-radius:10px; padding:22px;
                font-family: IBM Plex Mono, monospace;
                font-size:12px; color:#5a8c6a; line-height:2.2;'>
        DATASET &nbsp;&nbsp;&nbsp;→ &nbsp;Cleveland (303 patients)<br>
        SELECTION &nbsp;→ &nbsp;MI + Chi² → Top-8 of 13<br>
        TUNING &nbsp;&nbsp;&nbsp;→ &nbsp;GridSearchCV · 5-Fold AUC<br>
        MODELS &nbsp;&nbsp;&nbsp;→ &nbsp;KNN · LR · Gradient Boosting<br>
        DEPLOYED &nbsp;&nbsp;→ &nbsp;<span style='color:#34d399;'>{model_label}</span><br>
        INTERPRET &nbsp;→ &nbsp;SHAP · Permutation · PDP<br>
        STATUS &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;<span style='color:#34d399;'>{"● READY" if model_loaded else "● NOT LOADED"}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if model_loaded:
        run = st.button("🔍 RUN RISK ANALYSIS")
    else:
        run = False
        st.warning("Model not loaded — run Stages 5–7 notebooks first.")

with left:
    if run:
        with st.spinner("⚙️ Running diagnostic inference pipeline..."):
            time.sleep(1.2)

        # Full 13-feature row in training order
        ALL_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        raw_all    = pd.DataFrame(
            [[age, sex_val, cp, trestbps, chol, fbs,
              restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=ALL_FEATURES
        )
        scaled_all = scaler.transform(raw_all)
        scaled_df  = pd.DataFrame(scaled_all, columns=ALL_FEATURES)

        # Feed only Top-8 to model
        model_input = scaled_df[TOP_FEATURES].values
        prediction  = model.predict(model_input)

        try:
            prob      = model.predict_proba(model_input)[0]
            prob_pos  = round(prob[1] * 100, 1)
            prob_neg  = round(prob[0] * 100, 1)
            has_proba = True
        except Exception:
            has_proba = False

        st.markdown("### 📋 Diagnostic Report")

        if prediction[0] == 1:
            st.error("🚨  POSITIVE — CARDIAC RISK DETECTED")
            st.warning("⚠️  Immediate cardiology consultation advised based on these clinical markers.")
            if has_proba:
                st.markdown(f"""
                <div style='background:#1a0d0d; border:1px solid rgba(220,38,38,0.3);
                            border-radius:10px; padding:16px; margin-top:12px;
                            font-family: IBM Plex Mono, monospace; font-size:13px;'>
                    <span style='color:#f87171;'>RISK PROBABILITY &nbsp;: {prob_pos}%</span><br>
                    <span style='color:#5a8c6a;'>SAFE PROBABILITY &nbsp;: {prob_neg}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅  NEGATIVE — STABLE CARDIAC PROFILE")
            st.info("📋  All markers within acceptable safety ranges. Routine monitoring recommended.")
            if has_proba:
                st.markdown(f"""
                <div style='background:#0a1f10; border:1px solid rgba(52,211,153,0.3);
                            border-radius:10px; padding:16px; margin-top:12px;
                            font-family: IBM Plex Mono, monospace; font-size:13px;'>
                    <span style='color:#34d399;'>SAFE PROBABILITY &nbsp;&nbsp;: {prob_neg}%</span><br>
                    <span style='color:#f87171;'>RISK PROBABILITY &nbsp;: {prob_pos}%</span>
                </div>
                """, unsafe_allow_html=True)

            import random
            pulses = ""
            for _ in range(8):
                sz  = random.randint(20, 50)
                lft = random.randint(5, 90)
                tp  = random.randint(5, 90)
                pulses += f"""
                <div class='pulse-dot' style='
                    width:{sz}px; height:{sz}px; left:{lft}vw; top:{tp}vh;
                    animation-duration:{round(random.uniform(1.5,3),2)}s;
                    animation-delay:{round(random.uniform(0,3),2)}s;
                '></div>"""
            st.markdown(pulses, unsafe_allow_html=True)

        # Top-8 feature input table
        st.markdown("#### 📌 Model Input — Top-8 Features")
        top8_display = pd.DataFrame({
            "Feature": TOP_FEATURES,
            "Raw Value": [raw_all[f].values[0] for f in TOP_FEATURES],
            "Scaled Value": [round(scaled_df[f].values[0], 4) for f in TOP_FEATURES]
        })
        st.dataframe(top8_display, use_container_width=True, hide_index=True)
        st.markdown(
            "<p style='font-size:11px; color:#2d6645;'>"
            "* Features selected via combined Mutual Information + Chi-Square ranking (Stage 5). "
            "All 13 inputs are standardised; only these 8 are passed to the model.</p>",
            unsafe_allow_html=True
        )

    else:
        st.markdown("""
        <div style='background:#0f2018; border:1px dashed rgba(52,211,153,0.25);
                    border-radius:12px; padding:60px; text-align:center; margin-top:10px;'>
            <div style='font-size:44px;'>🫀</div>
            <p style='font-family: IBM Plex Mono, monospace; color:#34d399;
                      letter-spacing:2px; margin-top:14px;'>AWAITING PATIENT DATA...</p>
            <p style='font-size:11px; color:#2d6645; letter-spacing:1px;'>
                FILL IN PARAMETERS VIA THE SIDEBAR · THEN CLICK RUN RISK ANALYSIS
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. INTERPRETABILITY PANEL (Stage 8)
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("🧠 Stage 8 — Model Interpretability Summary (SHAP & Permutation Importance)"):
    st.markdown("""
    The deployed model was interpreted using three complementary techniques in **Stage 8**:

    | Technique | Scope | Key Finding |
    |-----------|-------|-------------|
    | **SHAP Beeswarm** | Global | `cp` and `thalach` dominate predictions |
    | **SHAP Waterfall** | Local (per patient) | High `oldpeak` + low `thalach` → elevated risk |
    | **Permutation Importance** | Global | Shuffling `thalach`, `ca`, `cp`, `thal` causes the largest AUC drop |
    | **Partial Dependence (PDP)** | Global | `thalach` has a clear negative trend — higher max HR → lower disease risk |

    These findings align with established cardiology literature and confirm the model makes
    **clinically interpretable decisions**.

    Plots saved to: `plots/shap_beeswarm.png` · `plots/shap_bar.png` ·
    `plots/shap_waterfall_disease.png` · `plots/permutation_importance.png` ·
    `plots/partial_dependence.png`
    """)

# ─────────────────────────────────────────────
# 8. ABOUT / DISCLAIMER
# ─────────────────────────────────────────────
with st.expander("ℹ️ About this Application & Disclaimer"):
    st.markdown("""
    **HeartGuard AI** is a clinical decision-support prototype built as part of the
    *Predictive Analytics* group project (Academic Year 2025–26).

    **Dataset:** Cleveland Heart Disease Dataset (UCI Repository) — 303 patients,
    13 clinical attributes, binary target (0 = No Disease, 1 = Disease).

    **Full Pipeline:**
    - **Stage 5:** MI + Chi-Square feature selection → Top-8 of 13 features
    - **Stage 6:** KNN, Logistic Regression, Gradient Boosting with GridSearchCV (5-fold, ROC-AUC)
    - **Stage 7:** 6 models evaluated (3 models × full + top-8); best model selected
    - **Stage 8:** SHAP (global + local), Permutation Importance, PDP
    - **Stage 9:** This Streamlit deployment

    **⚠️ Disclaimer:** For **educational and research purposes only**.
    Not a certified medical device. Must not substitute professional medical judgement.
    """)

# ─────────────────────────────────────────────
# 9. FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:10px; color:#2d6645; letter-spacing:2px; padding:4px 0 8px;'>
    🫀 &nbsp; HEARTGUARD AI v2.0 &nbsp;·&nbsp; PREDICTIVE ANALYTICS 2025-26
    &nbsp;·&nbsp; FOR RESEARCH USE ONLY
</div>
""", unsafe_allow_html=True)
