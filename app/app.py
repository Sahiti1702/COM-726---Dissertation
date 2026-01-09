
# APP/app.py  (FULL STREAMLIT)
# Hybrid Stroke Prediction GUI


import os
import datetime as dt
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from catboost import CatBoostClassifier, Pool
import plotly.express as px


# Page config

st.set_page_config(page_title="Stroke Prediction (Hybrid)", layout="wide")


# ✅ Clinical Dark Theme

st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(1200px 800px at 10% 10%, #172554 0%, #0b1020 55%, #070a14 100%);
  color: #EAF2FF;
}
html, body, [class*="css"]  { color: #EAF2FF !important; }
h1, h2, h3, h4 { color: #EAF2FF !important; letter-spacing: 0.2px; }
.small-muted { color: rgba(234,242,255,0.75); }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 700; font-size: 15px; color: rgba(234,242,255,0.85) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #EAF2FF !important; border-bottom: 3px solid #22c55e !important; }

/* Cards */
.card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 16px 18px; }
.card-title { font-weight: 800; font-size: 16px; margin-bottom: 6px; }
.card-sub { color: rgba(234,242,255,0.75); font-size: 13px; margin-top: -2px; }

/* Decision banners */
.banner { border-radius: 12px; padding: 14px 16px; border: 1px solid rgba(255,255,255,0.12); }
.banner-high { background: rgba(239,68,68,0.16); border-left: 6px solid #ef4444; }
.banner-low  { background: rgba(34,197,94,0.16); border-left: 6px solid #22c55e; }

/* Risk band */
.band { border-radius: 12px; padding: 12px 14px; border: 1px solid rgba(255,255,255,0.12); margin-top: 10px; }
.band-low      { background: rgba(34,197,94,0.14);  border-left: 6px solid #22c55e; }
.band-moderate { background: rgba(234,179,8,0.14);  border-left: 6px solid #eab308; }
.band-high     { background: rgba(249,115,22,0.14); border-left: 6px solid #f97316; }
.band-vhigh    { background: rgba(239,68,68,0.14);  border-left: 6px solid #ef4444; }

/* Next steps */
.steps { border-radius: 12px; padding: 14px 16px; border: 1px solid rgba(255,255,255,0.12); margin-top: 12px; }
.steps-low  { background: rgba(14,165,233,0.10); border-left: 6px solid rgba(14,165,233,0.85); }
.steps-high { background: rgba(239,68,68,0.10); border-left: 6px solid rgba(239,68,68,0.85); }
.steps-title { font-weight: 900; margin-bottom: 6px; }

/* Score rows */
.score-row { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
.score-label { font-weight: 800; color: rgba(234,242,255,0.92); }
.score-value { font-weight: 900; font-size: 18px; }

/* Buttons */
div.stButton > button {
  background: #22c55e; color: #061018 !important; font-weight: 800;
  border: 1px solid rgba(255,255,255,0.20); border-radius: 10px;
  padding: 0.70em 1.25em; box-shadow: none;
}
div.stButton > button:hover { background: #16a34a; border: 1px solid rgba(255,255,255,0.25); transform: none; }

/* SHAP section spacing */
.section-title { font-weight: 900; font-size: 18px; margin-top: 6px; }
.hr { height: 1px; background: rgba(255,255,255,0.12); margin: 14px 0; }
</style>
""",
    unsafe_allow_html=True,
)


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")

HYBRID_CFG_PATH = os.path.join(MODELS_DIR, "hybrid_config.pkl")

# -----------------------------
# Load assets (UNCHANGED)
# -----------------------------
@st.cache_resource
def load_hybrid_assets():
    cfg = joblib.load(HYBRID_CFG_PATH)

    cb = CatBoostClassifier()
    cb.load_model(cfg["cb_model_path"])

    cb_feature_names = joblib.load(cfg["cb_feature_names_path"])
    cb_cat_cols = joblib.load(cfg["cb_categorical_cols_path"])
    cb_cat_idx = [cb_feature_names.index(c) for c in cb_cat_cols if c in cb_feature_names]

    mlp_bundle = joblib.load(cfg["mlp_bundle_path"])
    pre = mlp_bundle["preprocessor"]
    mlp = mlp_bundle["model"]
    mlp_feature_names = mlp_bundle["feature_names"]
    mlp_cat_cols = mlp_bundle["categorical_cols"]
    mlp_num_cols = mlp_bundle["numeric_cols"]

    return {
        "cfg": cfg,
        "cb": cb,
        "cb_feature_names": cb_feature_names,
        "cb_cat_cols": cb_cat_cols,
        "cb_cat_idx": cb_cat_idx,
        "pre": pre,
        "mlp": mlp,
        "mlp_feature_names": mlp_feature_names,
        "mlp_cat_cols": mlp_cat_cols,
        "mlp_num_cols": mlp_num_cols,
    }

assets = load_hybrid_assets()

# -----------------------------
# Helpers (UNCHANGED logic)
# -----------------------------
def build_input_df(feature_names, values_dict):
    row = {col: np.nan for col in feature_names}
    for k, v in values_dict.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row])

def catboost_prob(cb_model, cb_feature_names, cb_cat_cols, cb_cat_idx, input_df):
    df_cb = input_df.reindex(columns=cb_feature_names, fill_value=np.nan).copy()
    for c in cb_cat_cols:
        if c in df_cb.columns:
            df_cb[c] = df_cb[c].fillna("Unknown").astype(str)
    for c in df_cb.columns:
        if c not in cb_cat_cols:
            df_cb[c] = pd.to_numeric(df_cb[c], errors="coerce").fillna(0)
    pool = Pool(df_cb, cat_features=cb_cat_idx)
    return float(cb_model.predict_proba(pool)[:, 1][0])

def mlp_prob(preprocessor, mlp_model, mlp_feature_names, mlp_cat_cols, mlp_num_cols, input_df):
    df_mlp = input_df.reindex(columns=mlp_feature_names, fill_value=np.nan).copy()
    for c in mlp_cat_cols:
        if c in df_mlp.columns:
            df_mlp[c] = df_mlp[c].fillna("Unknown").astype(str)
    for c in mlp_num_cols:
        if c in df_mlp.columns:
            df_mlp[c] = pd.to_numeric(df_mlp[c], errors="coerce").fillna(0)
    Xp = preprocessor.transform(df_mlp)
    return float(mlp_model.predict_proba(Xp)[:, 1][0])

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def get_risk_band(p_hybrid: float):
    if p_hybrid < 0.10:
        return "Low", "band-low", "Low estimated risk band (0.00–0.10)."
    elif p_hybrid < 0.30:
        return "Moderate", "band-moderate", "Moderate estimated risk band (0.10–0.30)."
    elif p_hybrid < 0.50:
        return "High", "band-high", "High estimated risk band (0.30–0.50)."
    else:
        return "Very High", "band-vhigh", "Very high estimated risk band (≥0.50)."

def next_steps_text(pred: int):
    if pred == 0:
        title = "Clinical Next Steps (Low Risk)"
        css = "steps steps-low"
        bullets = [
            "Encourage healthy lifestyle (balanced diet, physical activity, smoking avoidance).",
            "Monitor key indicators over time (blood pressure, glucose, weight/BMI).",
            "If symptoms occur (e.g., sudden weakness, slurred speech), seek urgent care immediately.",
            "This result is an estimate and should be interpreted alongside clinical judgement."
        ]
    else:
        title = "Clinical Next Steps (High Risk)"
        css = "steps steps-high"
        bullets = [
            "Recommend clinical review and further assessment by a healthcare professional.",
            "Consider confirming risk factors (blood pressure, diabetes screening, cholesterol profile).",
            "Discuss preventive actions (lifestyle modification, medication review if applicable).",
            "If any stroke warning signs are present, seek urgent medical attention immediately."
        ]
    return title, css, bullets

# Defaults / presets / reset

DEFAULTS = {
    "gender": "Male",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 120.0,
    "bmi": 25.0,
    "smoking_status": "never smoked",
}

DEMO_LOW = {
    "gender": "Female",
    "age": 28,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "No",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 92.0,
    "bmi": 21.5,
    "smoking_status": "never smoked",
}

DEMO_HIGH = {
    "gender": "Male",
    "age": 78,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Self-employed",
    "residence_type": "Rural",
    "avg_glucose_level": 220.0,
    "bmi": 33.0,
    "smoking_status": "smokes",
}

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def apply_preset(preset: dict):
    for k, v in preset.items():
        st.session_state[k] = v

def reset_inputs():
    apply_preset(DEFAULTS)

def init_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []

# ✅ validation-block toggle default
def init_validation_settings():
    if "block_extreme_inputs" not in st.session_state:
        st.session_state["block_extreme_inputs"] = True

init_state()
init_history()
init_validation_settings()


# Title

st.title("🧠 Stroke Risk Prediction Decision-Support System (Hybrid Model)")
st.markdown(
    '<div class="small-muted">Hybrid probability fusion using saved tuned weight & threshold. SHAP plots are loaded from the figures folder.</div>',
    unsafe_allow_html=True
)
st.write("")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Explainability (SHAP)", "ℹ️ About"])


# TAB 1 — Prediction

with tab1:
    st.markdown(
        '<div class="card"><div class="card-title">Patient Inputs</div>'
        '<div class="card-sub">Use presets for quick demos, or enter values manually. Then run the risk assessment.</div></div>',
        unsafe_allow_html=True
    )
    st.write("")

    b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.2, 3])
    with b1:
        if st.button("Demo: Low Risk", key="preset_low"):
            apply_preset(DEMO_LOW); st.rerun()
    with b2:
        if st.button("Demo: High Risk", key="preset_high"):
            apply_preset(DEMO_HIGH); st.rerun()
    with b3:
        if st.button("Reset Inputs", key="reset_btn"):
            reset_inputs(); st.rerun()
    with b4:
        st.caption("Tip: Use demo presets for screenshots/presentation, then reset to defaults.")

    # ✅ Toggle to block extreme inputs
    st.checkbox(
        "Prevent prediction when extreme inputs are detected (recommended for demos)",
        key="block_extreme_inputs"
    )

    st.write("")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                              index=["Male", "Female", "Other"].index(st.session_state["gender"]),
                              key="gender")
        age = st.slider("Age", 0, 100, int(st.session_state["age"]), key="age")
        hypertension = st.selectbox("Hypertension (0/1)", [0, 1],
                                    index=[0, 1].index(int(st.session_state["hypertension"])),
                                    key="hypertension")
        heart_disease = st.selectbox("Heart Disease (0/1)", [0, 1],
                                     index=[0, 1].index(int(st.session_state["heart_disease"])),
                                     key="heart_disease")

    with c2:
        ever_married = st.selectbox("Ever Married", ["Yes", "No"],
                                    index=["Yes", "No"].index(st.session_state["ever_married"]),
                                    key="ever_married")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                                 index=["Private", "Self-employed", "Govt_job", "children", "Never_worked"].index(st.session_state["work_type"]),
                                 key="work_type")
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"],
                                      index=["Urban", "Rural"].index(st.session_state["residence_type"]),
                                      key="residence_type")

    with c3:
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0,
                                            value=float(st.session_state["avg_glucose_level"]),
                                            step=1.0, key="avg_glucose_level")
        bmi = st.number_input("BMI", min_value=0.0,
                              value=float(st.session_state["bmi"]),
                              step=0.1, key="bmi")
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"],
                                      index=["never smoked", "formerly smoked", "smokes", "Unknown"].index(st.session_state["smoking_status"]),
                                      key="smoking_status")

    # -----------------------------
    # ✅ Input Validation + Extreme Detection (NEW)
    # -----------------------------
    warnings = []
    infos = []
    extreme_flags = []  # if any -> can block prediction when toggle enabled

    # BMI validation
    if bmi < 10 or bmi > 70:
        warnings.append("**BMI looks unusual** (expected roughly 10–70). Please verify the value.")
        extreme_flags.append("BMI outside 10–70")
    elif bmi < 15 or bmi > 50:
        infos.append("BMI is outside the typical adult range. Please confirm the input is correct.")

    # Glucose validation (mg/dL)
    if avg_glucose_level > 400:
        warnings.append("**Average glucose level is extremely high** (>400). Please verify the value and units (mg/dL).")
        extreme_flags.append("Glucose > 400")
    elif avg_glucose_level > 250:
        infos.append("Average glucose level is very high. Please verify the value and units (mg/dL).")
    elif avg_glucose_level < 40:
        warnings.append("**Average glucose level is extremely low** (<40). Please verify the value and units (mg/dL).")
        extreme_flags.append("Glucose < 40")

    # Age sanity
    if age > 100:
        infos.append("Age is very high. Please confirm the input is correct.")

    # Smoking unknown
    if smoking_status == "Unknown":
        infos.append("Smoking status is set to 'Unknown'. If known, selecting the correct status may improve interpretability.")

    if warnings or infos:
        with st.expander("Validation & data quality checks"):
            if warnings:
                for wmsg in warnings:
                    st.warning(wmsg)
            if infos:
                for imsg in infos:
                    st.info(imsg)
            st.caption("These checks do not change the prediction; they only help verify inputs for demo quality.")

    # Engineered feature
    age_group = None
    if "age_group" in assets["cb_feature_names"] or "age_group" in assets["mlp_feature_names"]:
        if age <= 18:
            age_group = "Child"
        elif age <= 40:
            age_group = "YoungAdult"
        elif age <= 65:
            age_group = "Adult"
        else:
            age_group = "Senior"
        st.info(f"Auto age_group used: **{age_group}** (based on age)")

    values = {
        "gender": gender,
        "age": int(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": float(bmi),
        "smoking_status": smoking_status,
    }
    if age_group is not None:
        values["age_group"] = age_group

    # ✅ Block logic (UI only, no model changes)
    can_predict = True
    if st.session_state["block_extreme_inputs"] and extreme_flags:
        can_predict = False
        st.error(
            "Prediction is temporarily blocked because **extreme inputs** were detected. "
            "Please verify inputs or disable the blocking option.\n\n"
            f"Detected: {', '.join(extreme_flags)}"
        )

    st.write("")
    if st.button("Run Risk Assessment", type="primary", disabled=not can_predict):
        input_df = build_input_df(assets["cb_feature_names"], values)

        p_cb = catboost_prob(
            assets["cb"], assets["cb_feature_names"],
            assets["cb_cat_cols"], assets["cb_cat_idx"],
            input_df
        )
        p_mlp = mlp_prob(
            assets["pre"], assets["mlp"],
            assets["mlp_feature_names"],
            assets["mlp_cat_cols"], assets["mlp_num_cols"],
            input_df
        )

        w = float(assets["cfg"]["weight_cb"])
        thr = float(assets["cfg"]["threshold"])
        p_hybrid = w * p_cb + (1 - w) * p_mlp
        pred = int(p_hybrid >= thr)

        timestamp = dt.datetime.now().isoformat(timespec="seconds")

        band, band_css, band_desc = get_risk_band(p_hybrid)

        # store to session history
        row = {"timestamp": timestamp, **values}
        row.update({
            "catboost_p": p_cb,
            "mlp_p": p_mlp,
            "hybrid_p": p_hybrid,
            "decision": "High Risk (1)" if pred == 1 else "Low Risk (0)",
            "risk_band": band,
            "weight_cb": w,
            "threshold": thr,
        })
        st.session_state["history"].append(row)

        # Results UI
        st.write("")
        st.markdown(
            '<div class="card"><div class="card-title">Prediction Results</div>'
            '<div class="card-sub">Probabilities are model estimates (not a diagnosis). For educational decision-support only.</div></div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([1.25, 1])

        with left:
            if pred == 1:
                st.markdown(
                    f"""<div class="banner banner-high">
<b>⚠️ High Risk (Positive at threshold)</b><br>
Hybrid P(stroke) = <b>{p_hybrid:.3f}</b> | Threshold = <b>{thr:.3f}</b>
</div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div class="banner banner-low">
<b>✅ Low Risk (Negative at threshold)</b><br>
Hybrid P(stroke) = <b>{p_hybrid:.3f}</b> | Threshold = <b>{thr:.3f}</b>
</div>""",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""<div class="band {band_css}">
<b>Risk Band: {band}</b><br>
{band_desc}<br>
<span class="small-muted">Risk band is interpretive only (not a diagnosis).</span>
</div>""",
                unsafe_allow_html=True
            )

            title, steps_css, bullets = next_steps_text(pred)
            bullet_html = "".join([f"<li>{b}</li>" for b in bullets])
            st.markdown(
                f"""
                <div class="{steps_css}">
                  <div class="steps-title">🩺 {title}</div>
                  <ul>{bullet_html}</ul>
                  <span class="small-muted">These suggestions are informational and should not be treated as medical advice.</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                '<div class="card"><div class="card-title">Model Breakdown</div>'
                '<div class="card-sub">Click ℹ️ to read what each probability represents.</div></div>',
                unsafe_allow_html=True
            )
            st.write("")

            tip_cb = f"CatBoost estimate: {pct(p_cb)} probability that the patient will have a stroke."
            tip_mlp = f"MLP estimate: {pct(p_mlp)} probability that the patient will have a stroke."
            tip_h = f"Hybrid estimate: {pct(p_hybrid)} probability that the patient will have a stroke (fusion of CatBoost and MLP)."

            r1a, r1b = st.columns([4, 1])
            with r1a:
                st.markdown(
                    f'<div class="score-row"><span class="score-label">CatBoost P(stroke)</span><br>'
                    f'<span class="score-value">{p_cb:.3f}</span></div>',
                    unsafe_allow_html=True
                )
            with r1b:
                with st.popover("ℹ️"):
                    st.write(tip_cb)

            r2a, r2b = st.columns([4, 1])
            with r2a:
                st.markdown(
                    f'<div class="score-row"><span class="score-label">MLP P(stroke)</span><br>'
                    f'<span class="score-value">{p_mlp:.3f}</span></div>',
                    unsafe_allow_html=True
                )
            with r2b:
                with st.popover("ℹ️"):
                    st.write(tip_mlp)

            r3a, r3b = st.columns([4, 1])
            with r3a:
                st.markdown(
                    f'<div class="score-row"><span class="score-label">Hybrid P(stroke)</span><br>'
                    f'<span class="score-value">{p_hybrid:.3f}</span></div>',
                    unsafe_allow_html=True
                )
            with r3b:
                with st.popover("ℹ️"):
                    st.write(tip_h)

        # Latest CSV download
        st.write("")
        st.markdown("### Download Latest Result (CSV)")
        csv_latest = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")
        st.download_button("Download Latest CSV", data=csv_latest, file_name="stroke_prediction_latest.csv", mime="text/csv")

        # Session history table + download
        st.write("")
        st.markdown("### Session History (This Run Only)")
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True)

        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Session History (CSV)",
            data=csv_hist,
            file_name="stroke_prediction_session_history.csv",
            mime="text/csv"
        )

        with st.expander("Clear session history"):
            st.warning("This will remove the history table for the current session only.")
            if st.button("Clear history now"):
                st.session_state["history"] = []
                st.success("Session history cleared.")
                st.rerun()


# TAB 2 — SHAP

with tab2:
    st.markdown('<div class="section-title">Explainability (SHAP) — CatBoost Component</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card">
  <div class="card-title">How to read SHAP</div>
  <div class="card-sub">
    • <b>Global summary</b> shows how features influence predictions across the dataset.<br>
    • <b>Feature importance</b> ranks features by average impact (mean |SHAP|).<br>
    • <b>Waterfall (local)</b> explains one individual prediction (feature contributions).
  </div>
</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    summary_img = os.path.join(FIG_DIR, "shap_summary_catboost.png")
    shap_csv = os.path.join(FIG_DIR, "shap_importance_catboost.csv")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Global Summary Plot (Static)")
        st.caption("Static image generated during your SHAP analysis step.")
        if os.path.exists(summary_img):
            st.image(summary_img, use_container_width=True)
        else:
            st.warning("shap_summary_catboost.png not found in figures/")

    with colB:
        st.markdown("### Global Feature Importance (Interactive)")
        st.caption("Hover the bars to view exact values (from shap_importance_catboost.csv).")
        if os.path.exists(shap_csv):
            fi = pd.read_csv(shap_csv).copy()
            cols_lower = {c.lower(): c for c in fi.columns}
            feat_col = cols_lower.get("feature", fi.columns[0])
            imp_col = fi.columns[1] if len(fi.columns) >= 2 else fi.columns[0]

            plot_df = fi[[feat_col, imp_col]].dropna().head(20).copy()
            plot_df.columns = ["Feature", "Importance"]

            fig = px.bar(plot_df[::-1], x="Importance", y="Feature", orientation="h",
                         hover_data={"Importance": ":.6f"})
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("shap_importance_catboost.csv not found in figures/")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Local Explanation (Waterfall Plot)")
    st.caption("Select a saved waterfall plot image for an individual instance.")

    waterfall_files = sorted([
        f for f in os.listdir(FIG_DIR)
        if f.startswith("shap_waterfall_catboost_") and f.endswith(".png")
    ]) if os.path.exists(FIG_DIR) else []

    if waterfall_files:
        selected = st.selectbox("Choose a waterfall plot", waterfall_files)
        st.image(os.path.join(FIG_DIR, selected), use_container_width=True)
        st.markdown(
            """
**How to interpret this waterfall plot:**
- Bars pushing the prediction **up** increase the estimated risk for this case.
- Bars pushing the prediction **down** reduce the estimated risk for this case.
- The final output = **baseline + sum of feature contributions** (CatBoost component).
            """
        )
    else:
        st.info("No waterfall images found. Expected: shap_waterfall_catboost_*.png inside figures/.")

    with st.expander("Show SHAP top-features table (from CSV)"):
        if os.path.exists(shap_csv):
            st.dataframe(pd.read_csv(shap_csv).head(30), use_container_width=True)
        else:
            st.warning("shap_importance_catboost.csv not found in figures/")

# =============================
# TAB 3 — About
# =============================
with tab3:
    st.subheader("About this Decision-Support System")

    st.markdown(
        """
**Project Title:**  
*Stroke Risk Prediction Decision-Support System using a Hybrid Model*

**Programme:** MSc Applied AI & Data Science (COM726 Dissertation)

---

### Purpose of the System
This application demonstrates an **AI-based decision-support system** designed to estimate the **risk of stroke** using routinely collected patient information.
The goal of the system is to **support interpretation and discussion**, not to provide a medical diagnosis.

---

### Modelling Approach
The system uses a **hybrid machine learning framework**, combining:
- **CatBoost** — effective for structured clinical data and categorical features  
- **Multilayer Perceptron (MLP)** — capable of capturing non-linear relationships  

The final prediction is obtained via **probability-level fusion** using a tuned weight and decision threshold selected during evaluation.
This approach aims to balance **robustness, interpretability, and predictive performance**.

---

### Explainability and Transparency
To improve interpretability:
- **SHAP (SHapley Additive exPlanations)** is used to explain model behaviour
- Global plots highlight important risk factors across the dataset
- Local waterfall plots explain individual predictions

Risk bands and clinical-style explanations are included to improve usability for non-technical stakeholders.

---

### Intended Use and Limitations
- This system is intended for **educational and research purposes only**
- Predictions represent **statistical risk estimates**, not diagnoses
- Outputs should always be interpreted **alongside clinical expertise**
- The system does not replace professional medical assessment

---

### Ethical Considerations
The application follows principles of responsible AI by:
- Avoiding automated clinical decisions
- Providing transparent explanations
- Including clear disclaimers
- Preventing over-reliance on model outputs

---

**Disclaimer:**  
This tool is not a medical device and should not be used for real-world clinical decision-making.
        """
    )

    st.markdown("**How to run the application locally:**")
    st.code(
        """
cd COM726_Stroke_Dissertation/App
streamlit run app.py
        """.strip()
    )
