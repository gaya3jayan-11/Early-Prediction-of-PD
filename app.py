# app.py  — Minimal Streamlit UI for batch CSV scoring
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
PLOTS_DIR = Path("plots")
DATA_DIR = Path("data/processed")

SCALER_PATH = MODELS_DIR / "scaler.pkl"
PIPE_PATH = MODELS_DIR / "Best_Model_pipeline.pkl"
FEATS_PATH = MODELS_DIR / "feature_names.json"
MANIFEST_PATH = MODELS_DIR / "dataset_manifest.json"
QUICK_PATH = REPORTS_DIR / "quick_checks.json"
CALIB_PLOT = PLOTS_DIR / "calibration_curve.png"

st.set_page_config(page_title="PD Risk Batch Scorer", layout="wide")

# --------------------------
# Cached loaders
# --------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    pipe = joblib.load(PIPE_PATH)
    if FEATS_PATH.exists():
        features = json.loads(FEATS_PATH.read_text())
    else:
        # Fallback (shouldn't be needed): infer from data
        sample = pd.read_csv(DATA_DIR / "final_dataset_enhanced.csv", nrows=5)
        features = [c for c in sample.columns if c not in ("convert_84m", "PATNO")]
    return scaler, pipe, features

@st.cache_data
def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return None

@st.cache_data
def load_quick_checks():
    if QUICK_PATH.exists():
        return json.loads(QUICK_PATH.read_text())
    return None

# --------------------------
# Sidebar: model & dataset info
# --------------------------
st.sidebar.title("Model & Data")
exists = {
    "scaler.pkl": SCALER_PATH.exists(),
    "Best_Model_pipeline.pkl": PIPE_PATH.exists(),
    "feature_names.json": FEATS_PATH.exists(),
}
for k, ok in exists.items():
    st.sidebar.write(("✅ " if ok else "❌ ") + k)

manifest = load_manifest()
if manifest:
    st.sidebar.markdown("**Dataset manifest**")
    st.sidebar.json(
        {k: manifest[k] for k in ["rows", "cols", "positives", "pos_rate", "sha256"] if k in manifest},
        expanded=False,
    )

qc = load_quick_checks()
if qc:
    st.sidebar.markdown("**Quick checks (full CSV)**")
    st.sidebar.json(
        {k: qc[k] for k in ["n", "positives", "pos_rate", "auroc", "auprc", "brier", "ece_10q"] if k in qc},
        expanded=False,
    )
    if CALIB_PLOT.exists():
        st.sidebar.image(str(CALIB_PLOT), caption="Calibration curve", use_container_width=True)

# --------------------------
# Main UI
# --------------------------
st.title("PD Risk Batch Scorer (CSV)")

with st.expander("Instructions", expanded=False):
    st.markdown(
        "- Upload a CSV with the same columns used in training (the app will align them automatically).\n"
        "- Optionally, tick **Use sample (final_dataset_enhanced.csv)** to demo on your dataset."
    )

scaler, pipe, features = load_artifacts()

col_left, col_right = st.columns([2, 1])
with col_right:
    thr = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

with col_left:
    use_sample = st.checkbox("Use sample (data/processed/final_dataset_enhanced.csv)")
    file = None
    if not use_sample:
        file = st.file_uploader("Upload CSV", type=["csv"])

run_btn = st.button("Run prediction")

# --------------------------
# Helper: align features
# --------------------------
def align_features(df: pd.DataFrame, expected: list[str]) -> tuple[pd.DataFrame, list[str]]:
    # Reindex to expected columns (adds any missing as NaN, and orders correctly)
    X = df.reindex(columns=expected)
    missing = [c for c in expected if c not in df.columns]
    return X, missing

# --------------------------
# Inference
# --------------------------
if run_btn:
    try:
        if use_sample:
            src = DATA_DIR / "final_dataset_enhanced.csv"
            if not src.exists():
                st.error(f"Sample file not found: {src}")
                st.stop()
            df = pd.read_csv(src, low_memory=False)
            st.info(f"Loaded sample: {src} — shape={df.shape}")
        else:
            if file is None:
                st.warning("Please upload a CSV or enable 'Use sample'.")
                st.stop()
            df = pd.read_csv(file, low_memory=False)
            st.info(f"Uploaded file shape: {df.shape}")

        # Keep optional ID column for convenience if present
        id_col = None
        for cand in ["PATNO", "subject_id", "ID", "id"]:
            if cand in df.columns:
                id_col = cand
                break

        # Align to training features (exact order; add missing as NaN)
        X, missing = align_features(df, features)
        if len(missing) > 0:
            st.warning(f"{len(missing)} expected feature(s) were missing and set to NaN. Example: {missing[:8]}")

        # Use DataFrame so sklearn keeps feature names (avoids warnings)
        Xs = scaler.transform(X)

        proba = pipe.predict_proba(Xs)[:, 1]
        yhat = (proba >= thr).astype(int)

        out = pd.DataFrame(index=df.index)
        if id_col:
            out[id_col] = df[id_col]
        out["probability"] = proba
        out["prediction"] = yhat

        # Quick summary
        pos = int(yhat.sum())
        st.success(f"Predicted positives at threshold {thr:.2f}: {pos}/{len(out)}")

        # Show preview
        st.dataframe(out.head(50), use_container_width=True)

        # Download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.exception(e)
