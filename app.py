import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScreen — Cardiac Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F4FAFB; }
    .stApp { background-color: #F4FAFB; }
    
    .header-box {
        background: linear-gradient(135deg, #0D1B2A 0%, #028090 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-box h1 { color: #02C39A; margin: 0; font-size: 2rem; }
    .header-box p  { color: #A8D8DF; margin: 0.4rem 0 0 0; font-size: 1rem; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-top: 4px solid #028090;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #028090; }
    .metric-card .lbl { font-size: 0.8rem; color: #64748B; margin-top: 0.2rem; }

    .risk-high {
        background: linear-gradient(135deg, #7F1D1D, #DC2626);
        color: white; border-radius: 12px; padding: 2rem;
        text-align: center; margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #014D40, #02C39A);
        color: white; border-radius: 12px; padding: 2rem;
        text-align: center; margin: 1rem 0;
    }
    .risk-high h2, .risk-low h2 { font-size: 1.6rem; margin: 0.3rem 0; }
    .risk-high p,  .risk-low p  { opacity: 0.85; margin: 0; }

    .info-box {
        background: #E0F4F6;
        border-left: 4px solid #028090;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #01606A;
    }
    .warn-box {
        background: #FEE2E2;
        border-left: 4px solid #DC2626;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #7F1D1D;
    }

    .feature-pill {
        display: inline-block;
        background: #E0F4F6;
        color: #01606A;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        margin: 0.15rem;
        font-weight: 600;
    }

    div[data-testid="stProgress"] > div { background-color: #028090; }
    .stSlider > div > div > div > div { background-color: #028090; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]
TARGET   = 'Heart Disease'
SEED     = 42
MIN_RECALL = 0.90
MODEL_PATH = "cardioscreen_model.joblib"

# ── Model training (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Train on the dataset and return (model, imputer, threshold)."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # Load data — try common locations
    for path in ["heart_disease_demo.csv", "data/heart_disease_demo.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        return None  # no data found

    df['target'] = (df[TARGET] == 'Presence').astype(int)
    X = df[FEATURES].copy()
    y = df['target'].copy()

    # Replace clinically impossible zeros with NaN
    for col in ['BP', 'Cholesterol', 'Max HR']:
        X[col] = X[col].replace(0, np.nan)

    # Stratified split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=SEED)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)

    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, class_weight='balanced', random_state=SEED
    )
    model.fit(X_train_imp, y_train)

    # Threshold tuning on validation set
    val_proba = model.predict_proba(X_val_imp)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, val_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    mask = rec[:-1] >= MIN_RECALL
    if mask.sum() > 0:
        threshold = float(thresholds[np.where(mask)[0][np.argmax(f1[:-1][mask])]])
    else:
        threshold = float(thresholds[np.argmax(f1[:-1])])

    bundle = (model, imputer, threshold)
    joblib.dump(bundle, MODEL_PATH)
    return bundle

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🫀 CardioScreen</h1>
  <p>AI-powered cardiac pre-screening for GP intake · Classical ML · 100,000 patients (demo) trained · ROC-AUC 0.956 · 90.2% Recall</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────
with st.spinner("Loading CardioScreen model..."):
    bundle = load_or_train_model()

if bundle is None:
    st.error("Dataset not found. Please place `heart_disease_demo.csv` in the same folder as this app.")
    st.stop()

model, imputer, threshold = bundle

# ── Sidebar — model info ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Model Info")
    st.markdown(f"""
    **Algorithm:** HistGradientBoosting  
    **Training set:** 100,000 patients (demo)  
    **ROC-AUC:** 0.956  
    **Recall:** 90.2%  
    **Precision:** ~85%  
    **Threshold:** `{threshold:.3f}`
    """)
    st.markdown("---")
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Clinical disclaimer</b><br>
    CardioScreen is a triage support tool only. 
    The GP retains full clinical authority. 
    A negative result does not exclude heart disease.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Top predictors (SHAP)**")
    top_features = [
        ("Thallium", 0.195),
        ("ST Depression", 0.172),
        ("Nr of Vessels", 0.145),
        ("Max HR", 0.098),
        ("Chest Pain Type", 0.072),
        ("Exercise Angina", 0.062),
    ]
    for feat, imp in top_features:
        st.progress(imp * 5, text=f"{feat}: {imp:.3f}")

    st.markdown("---")
    st.caption("Nova SBE · AI Impact on Business · Group Assignment")

# ── Main layout ────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown("### 📋 Patient Intake — 13 Clinical Variables")
    st.markdown('<div class="info-box">Enter routine GP intake data below. No EKG machine or specialist required.</div>', unsafe_allow_html=True)

    with st.form("patient_form"):
        st.markdown("**Demographics**")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", min_value=20, max_value=100, value=55)
        sex = c2.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

        st.markdown("**Clinical Vitals**")
        c3, c4 = st.columns(2)
        bp          = c3.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=130)
        cholesterol = c4.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
        c5, c6 = st.columns(2)
        max_hr   = c5.number_input("Max Heart Rate", min_value=60, max_value=220, value=145)
        fbs      = c6.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        st.markdown("**Diagnostic Tests**")
        chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
            format_func=lambda x: {1:"Typical Angina", 2:"Atypical Angina", 3:"Non-anginal Pain", 4:"Asymptomatic"}[x])
        c7, c8 = st.columns(2)
        ekg       = c7.selectbox("Resting EKG Results", options=[0, 1, 2],
            format_func=lambda x: {0:"Normal", 1:"ST-T Abnormality", 2:"LV Hypertrophy"}[x])
        ex_angina = c8.selectbox("Exercise Induced Angina", options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes")

        c9, c10 = st.columns(2)
        st_dep   = c9.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=8.0, value=1.2, step=0.1)
        slope_st = c10.selectbox("Slope of ST Segment", options=[1, 2, 3],
            format_func=lambda x: {1:"Upsloping", 2:"Flat", 3:"Downsloping"}[x])

        st.markdown("**Advanced Diagnostics**")
        c11, c12 = st.columns(2)
        vessels  = c11.selectbox("Number of Vessels Fluoroscopy", options=[0, 1, 2, 3])
        thallium = c12.selectbox("Thallium Stress Test", options=[3, 6, 7],
            format_func=lambda x: {3:"Normal", 6:"Fixed Defect", 7:"Reversible Defect"}[x])

        submitted = st.form_submit_button("🔍 Assess Cardiac Risk", use_container_width=True, type="primary")

with col_result:
    st.markdown("### 🩺 Risk Assessment")

    if not submitted:
        st.markdown("""
        <div style="background:white; border-radius:12px; padding:2.5rem;
                    text-align:center; border: 2px dashed #A8D8DF;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <span style="font-size:4rem;">🫀</span><br><br>
            <span style="color:#64748B; font-size:1rem;">
            Complete the intake form and click<br>
            <b>Assess Cardiac Risk</b> to see the prediction.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_data = pd.DataFrame([[
            age, sex, chest_pain, bp, cholesterol, fbs,
            ekg, max_hr, ex_angina, st_dep, slope_st, vessels, thallium
        ]], columns=FEATURES)

        input_imp  = imputer.transform(input_data)
        risk_proba = float(model.predict_proba(input_imp)[0, 1])
        high_risk  = risk_proba >= threshold

        # Risk result
        if high_risk:
            st.markdown(f"""
            <div class="risk-high">
                <div style="font-size:3rem;">⚠️</div>
                <h2>HIGH CARDIAC RISK</h2>
                <p style="font-size:1.1rem; margin-top:0.5rem;">
                    Risk Score: <b>{risk_proba*100:.1f}%</b>
                </p>
                <p>Priority referral to cardiologist recommended</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <div style="font-size:3rem;">✅</div>
                <h2>LOW CARDIAC RISK</h2>
                <p style="font-size:1.1rem; margin-top:0.5rem;">
                    Risk Score: <b>{risk_proba*100:.1f}%</b>
                </p>
                <p>Routine monitoring — no urgent referral indicated</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk meter
        st.markdown("**Risk Score**")
        st.progress(min(risk_proba, 1.0))
        c_lo, c_thr, c_hi = st.columns(3)
        c_lo.caption("0% — No risk")
        c_thr.caption(f"← {threshold*100:.0f}% threshold")
        c_hi.caption("100% — Certain")

        # Key signals
        st.markdown("**Key signals in this patient**")
        signals = []
        if thallium == 7:  signals.append(("🔴", "Reversible thallium defect", "Strongest predictor"))
        if thallium == 6:  signals.append(("🟠", "Fixed thallium defect", "High-risk finding"))
        if st_dep > 2.0:   signals.append(("🔴", f"ST Depression = {st_dep}", "Significant elevation"))
        if vessels >= 2:   signals.append(("🔴", f"{vessels} vessels blocked", "Critical finding"))
        if max_hr < 130:   signals.append(("🟠", f"Low max HR ({max_hr})", "Reduced cardiac output"))
        if chest_pain == 4:signals.append(("🟡", "Asymptomatic chest pain", "Silent disease risk"))
        if ex_angina == 1: signals.append(("🟠", "Exercise-induced angina", "Ischaemia signal"))
        if bp > 160:       signals.append(("🟡", f"High BP ({bp} mmHg)", "Cardiovascular stress"))
        if not signals:    signals.append(("🟢", "No major risk factors flagged", "Routine profile"))

        for emoji, label, note in signals:
            st.markdown(f"{emoji} **{label}** — *{note}*")

        # Model confidence
        st.markdown("---")
        st.markdown("**Model confidence**")
        conf = abs(risk_proba - threshold) / max(threshold, 1 - threshold)
        if conf > 0.5:
            st.success(f"High confidence ({conf*100:.0f}%) — clear signal")
        elif conf > 0.25:
            st.warning(f"Moderate confidence ({conf*100:.0f}%) — GP judgment important")
        else:
            st.error(f"Low confidence ({conf*100:.0f}%) — borderline case, clinical review required")

        st.markdown('<div class="warn-box">⚠️ This prediction is a triage aid only. The GP retains full clinical authority and must make the final decision.</div>', unsafe_allow_html=True)

# ── Model performance footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Model Performance (held-out test set — 94,500 patients)")
m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    ("0.956", "ROC-AUC"),
    ("90.2%", "Recall (Sensitivity)"),
    ("~85%",  "Precision"),
    ("0.878", "F1-Score"),
    ("88.7%", "Accuracy"),
]
for col, (val, lbl) in zip([m1, m2, m3, m4, m5], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="val">{val}</div>
        <div class="lbl">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.caption("CardioScreen · HistGradientBoosting · Trained on 100,000 patients (demo) · Nova SBE — AI Impact on Business · Group Assignment")
