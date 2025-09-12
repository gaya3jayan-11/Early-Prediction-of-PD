import json
import sqlite3
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load

# --- Optional: SHAP for interpretability (graceful fallback if not installed)
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# =============================
# App Config
# =============================
st.set_page_config(
    page_title="PD Risk Prediction – Clinical Preview",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
PLOTS_DIR = Path("plots")
DATA_DIR = Path("data/processed")

PIPE_PATH = MODELS_DIR / "Best_Model_pipeline.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATS_PATH = MODELS_DIR / "feature_names.json"
DATASET_CSV = DATA_DIR / "final_dataset_enhanced.csv"  # used for medians & template only

REMINDERS_DB = Path("data/reminders.db")
REMINDERS_DB.parent.mkdir(parents=True, exist_ok=True)

# =============================
# Helpers (cached)
# =============================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load scaler, pipeline, and feature list."""
    if not PIPE_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found in ./models. Please run training first.")

    pipe = load(PIPE_PATH)
    scaler = load(SCALER_PATH)

    if FEATS_PATH.exists():
        features = json.loads(FEATS_PATH.read_text())
    else:
        # Fallback: infer from dataset
        if not DATASET_CSV.exists():
            raise FileNotFoundError("feature_names.json missing and dataset not found to infer features.")
        df_tmp = pd.read_csv(DATASET_CSV, nrows=5)
        features = [c for c in df_tmp.columns if c not in ("convert_84m", "PATNO")]

    # pull underlying estimator for SHAP (if available)
    model = None
    try:
        model = pipe.named_steps.get("model", None)
    except Exception:
        model = None

    return scaler, pipe, features, model


@st.cache_data(show_spinner=False)
def training_statistics(features):
    """Compute training medians (for imputation) and provide a blank template."""
    med = pd.Series(dtype=float)
    template = pd.DataFrame(columns=features)
    if DATASET_CSV.exists():
        df = pd.read_csv(DATASET_CSV, low_memory=False)
        available = [c for c in features if c in df.columns]
        # Coerce numerics for robust scaling
        df_num = df[available].apply(pd.to_numeric, errors="coerce")
        med = df_num.median(numeric_only=True)
        # Fill missing medians with 0 to ensure no NaN pass through scaler
        med = med.reindex(features).fillna(0.0)
        template = pd.DataFrame(columns=features)
    else:
        # If dataset isn't present, default to zeros
        med = pd.Series(0.0, index=features)
        template = pd.DataFrame(columns=features)
    return med, template


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric, setting non-numeric to NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def preprocess_for_model(df_in: pd.DataFrame, features: list[str], medians: pd.Series, scaler) -> tuple[pd.DataFrame, np.ndarray]:
    """Align columns, coerce numeric, impute medians, scale."""
    X = df_in.reindex(columns=features)
    X = coerce_numeric_df(X)
    # Ensure medians index covers all features
    fill_series = medians.reindex(features).fillna(0.0)
    X = X.fillna(fill_series)
    # pass DataFrame to preserve feature names (prevents sklearn warning)
    Xs = scaler.transform(X)
    return X, Xs


def predict_dataframe(df_in: pd.DataFrame, features: list[str], medians: pd.Series, scaler, pipe) -> pd.DataFrame:
    X, Xs = preprocess_for_model(df_in, features, medians, scaler)
    proba = pipe.predict_proba(Xs)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df_in.copy()
    out["prediction"] = pred
    out["probability"] = proba
    return out


@st.cache_resource(show_spinner=False)
def shap_explainer(_model):
    """Cacheable SHAP explainer; leading underscore avoids Streamlit hashing."""
    if not SHAP_AVAILABLE or _model is None:
        return None
    try:
        # fast path for tree models (RF/XGB)
        return shap.TreeExplainer(_model)
    except Exception:
        return None


# =============================
# Reminders (SQLite)
# =============================
def init_db():
    with sqlite3.connect(REMINDERS_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                patient_id TEXT,
                title TEXT NOT NULL,
                due_date TEXT NOT NULL,
                notes TEXT,
                status TEXT NOT NULL DEFAULT 'scheduled'
            )
            """
        )
        conn.commit()


def add_reminder(patient_id: str, title: str, due_date: date, notes: str = ""):
    with sqlite3.connect(REMINDERS_DB) as conn:
        conn.execute(
            "INSERT INTO reminders (created_at, patient_id, title, due_date, notes, status) VALUES (?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                patient_id.strip() if patient_id else None,
                title.strip(),
                due_date.isoformat(),
                notes.strip() if notes else None,
                "scheduled",
            ),
        )
        conn.commit()


def list_reminders(status: str | None = None) -> pd.DataFrame:
    with sqlite3.connect(REMINDERS_DB) as conn:
        if status:
            df = pd.read_sql_query(
                "SELECT * FROM reminders WHERE status = ? ORDER BY due_date ASC, id DESC",
                conn,
                params=(status,),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM reminders ORDER BY due_date ASC, id DESC",
                conn,
            )
    return df


def update_status(reminder_id: int, new_status: str):
    with sqlite3.connect(REMINDERS_DB) as conn:
        conn.execute("UPDATE reminders SET status = ? WHERE id = ?", (new_status, reminder_id))
        conn.commit()


# =============================
# UI Sections
# =============================
def sidebar_nav() -> str:
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Predict", "Dashboard", "Reminders", "About"],
        index=0,
    )


def section_predict():
    st.header("Clinical Prediction & Explainability")

    # Load artifacts
    try:
        scaler, pipe, features, model = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return

    medians, template = training_statistics(features)

    colL, colR = st.columns([2, 1])
    with colL:
        st.subheader("Batch CSV Prediction")
        st.caption("Upload a CSV containing the model features. We'll align columns automatically.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                # preserve PATNO if present for convenience
                patno = df_in.get("PATNO") if "PATNO" in df_in.columns else None
                preds = predict_dataframe(df_in, features, medians, scaler, pipe)
                if patno is not None:
                    preds.insert(0, "PATNO", patno)
                st.success("Predictions computed.")
                st.dataframe(preds.head(50))

                # Download link
                csv_bytes = preds.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                # Optional: per-patient explanation (SHAP)
                if SHAP_AVAILABLE and model is not None:
                    st.markdown("---")
                    st.subheader("Per-patient explanation (SHAP)")
                    # Build an aligned, preprocessed matrix again for SHAP
                    X, Xs = preprocess_for_model(df_in, features, medians, scaler)

                    # Labels for selector
                    index_labels = (
                        patno.astype(str).values if patno is not None else X.index.astype(str).values
                    )

                    sel = st.selectbox(
                        "Select a row to explain",
                        options=list(range(len(X))),
                        format_func=lambda i: (
                            f"Row {i} (PATNO={index_labels[i]})" if i < len(index_labels) else f"Row {i}"
                        ),
                    )

                    try:
                        expl = shap_explainer(model)
                        if expl is None:
                            st.info("SHAP explainer not available for this model.")
                        else:
                            # Bound check
                            if sel < 0 or sel >= len(X):
                                st.warning("Selected row is out of range.")
                            else:
                                # Compute SHAP for the single row (1 x d)
                                row = Xs[sel : sel + 1]
                                sv_raw = expl.shap_values(row, check_additivity=False)

                                # For binary classifiers, shap_values can be list [neg, pos]
                                if isinstance(sv_raw, list) and len(sv_raw) == 2:
                                    sv = np.array(sv_raw[1]).reshape(-1)
                                else:
                                    sv = np.array(sv_raw).reshape(-1)

                                # Safety: align vector length to features if needed
                                if len(sv) != len(features):
                                    sv = sv[: len(features)]

                                # Plot top features manually to avoid JS dependencies
                                topk = min(15, len(features))
                                order = np.argsort(np.abs(sv))[::-1][:topk]
                                top_names = [features[i] for i in order][::-1]
                                top_vals = sv[order][::-1]

                                fig, ax = plt.subplots(figsize=(6, 5))
                                ax.barh(top_names, top_vals)
                                ax.set_xlabel("SHAP value (impact on log-odds)")
                                ax.set_title("Top feature contributions")
                                st.pyplot(fig, clear_figure=True)
                    except Exception as ex:
                        st.warning(f"Could not compute SHAP explanation: {ex}")
                else:
                    st.info("Install the 'shap' package to enable per-patient explanations.")

            except Exception as ex:
                st.error(f"Failed to run predictions: {ex}")

    with colR:
        st.subheader("Template & Feature Schema")
        st.caption("Use this template to prepare your batch CSV.")
        # Provide a one-row template
        one_row = pd.DataFrame({c: [0] for c in features})
        st.dataframe(one_row)
        st.download_button(
            "Download template CSV",
            data=one_row.to_csv(index=False).encode("utf-8"),
            file_name="pd_predict_template.csv",
            mime="text/csv",
        )
        st.markdown("**Model Inputs (feature_names.json):**")
        st.code("\n".join(features), language="text")

    st.markdown("---")
    st.caption(
        "This tool is a research decision-support aid and not a clinical diagnosis. Use alongside professional judgment."
    )


def section_dashboard():
    st.header("Model Performance & Monitoring")

    # Quick metrics
    metrics_path = REPORTS_DIR / "quick_checks.json"
    subgroup_path = REPORTS_DIR / "subgroup_metrics.csv"
    calib_img = PLOTS_DIR / "calibration_curve.png"

    cols = st.columns(4)

    try:
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        if metrics is None:
            st.warning("No reports/quick_checks.json found. Run training first.")
            return
        # Headline metrics
        cols[0].metric("Samples (n)", f"{metrics.get('n', '-')}")
        cols[1].metric("Positives", f"{metrics.get('positives', '-')}")
        cols[2].metric("AUROC", f"{metrics.get('auroc', 0):.3f}")
        cols[3].metric("AUPRC", f"{metrics.get('auprc', 0):.3f}")

        c2 = st.columns(3)
        c2[0].metric("Brier score", f"{metrics.get('brier', 0):.4f}")
        c2[1].metric("ECE (10q)", f"{metrics.get('ece_10q', 0):.4f}")
        c2[2].metric("Pos rate", f"{metrics.get('pos_rate', 0):.3%}")

        # Show which evaluation these metrics represent if provided (e.g., OOF / holdout)
        if "eval" in metrics:
            st.caption(f"Evaluation: {metrics['eval']}")

        st.markdown("### Calibration")
        if calib_img.exists():
            st.image(str(calib_img), caption="Reliability curve (reports)")
        else:
            st.info("Calibration plot not found at plots/calibration_curve.png")

        # Subgroups
        st.markdown("### Subgroup metrics")
        if subgroup_path.exists():
            sg = pd.read_csv(subgroup_path, index_col=0)
            st.dataframe(sg)
        else:
            st.info("No subgroup_metrics.csv available.")

        # Dataset manifest (optional)
        manifest_path = MODELS_DIR / "dataset_manifest.json"
        if manifest_path.exists():
            st.markdown("### Dataset Manifest")
            st.json(json.loads(manifest_path.read_text()))

    except Exception as e:
        st.error(f"Failed to load dashboard artifacts: {e}")


def section_reminders():
    st.header("Medical Reminders (Local)")
    init_db()

    with st.form("add_reminder"):
        st.subheader("Create a reminder")
        cols = st.columns(2)
        patient_id = cols[0].text_input("Patient ID (optional)")
        title = cols[1].text_input("Title (e.g., Repeat UPSIT)")
        due = st.date_input("Due date", value=date.today())
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Add reminder")
        if submitted:
            if not title.strip():
                st.warning("Title is required.")
            else:
                add_reminder(patient_id, title, due, notes)
                st.success("Reminder added.")

    st.markdown("---")
    st.subheader("Upcoming & Past reminders")

    tabs = st.tabs(["Scheduled", "Completed", "All"])
    status_filters = ["scheduled", "completed", None]

    for tab, filt in zip(tabs, status_filters):
        with tab:
            df = list_reminders(filt)
            if df.empty:
                st.info("No reminders to show.")
            else:
                st.dataframe(df)
                # Bulk actions
                colA, colB = st.columns(2)
                with colA:
                    sel_to_complete = st.multiselect(
                        "Select IDs to mark completed",
                        options=df["id"].tolist(),
                    )
                    if st.button("Mark completed", key=f"done_{filt}") and sel_to_complete:
                        for rid in sel_to_complete:
                            update_status(int(rid), "completed")
                        st.experimental_rerun()
                with colB:
                    sel_to_sched = st.multiselect(
                        "Select IDs to mark scheduled",
                        options=df["id"].tolist(),
                        key=f"sched_{filt}",
                    )
                    if st.button("Mark scheduled", key=f"sched_btn_{filt}") and sel_to_sched:
                        for rid in sel_to_sched:
                            update_status(int(rid), "scheduled")
                        st.experimental_rerun()

    st.caption("Reminders are stored locally in data/reminders.db (SQLite).")


def section_about():
    st.header("About & Disclaimers")
    st.markdown(
        """
        **Purpose:** This application provides a research preview of a Parkinson's Disease (PD) risk prediction tool.\
        It uses a trained machine learning model to estimate 7-year conversion risk based on structured features.\
        \n
        **Interpretability:** Per-patient explanations (when enabled) highlight feature contributions to the risk estimate.\
        \n
        **Important:** This is **not** a medical device or a diagnostic tool. Results should be considered alongside\
        clinical judgment and additional testing.
        """
    )


# =============================
# Main
# =============================
page = sidebar_nav()

if page == "Predict":
    section_predict()
elif page == "Dashboard":
    section_dashboard()
elif page == "Reminders":
    section_reminders()
else:
    section_about()
