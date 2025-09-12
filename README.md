## 🧠 Project Overview (What we have right now)

**Goal:** A research-grade tool that estimates a person’s 7-year risk of developing Parkinson’s Disease (PD) using routine, non-invasive signals (plasma biomarkers + simple clinical questionnaires).

### 🔎 Data we use (merged into one table)
- **Plasma biomarkers:** Lab-measured molecular signals from blood.
- **Smell test (UPSIT):** Odor identification score (hyposmia is a strong prodromal marker).
- **RBD/sleep questionnaire:** Self-reported dream-enactment/REM-sleep behavior.
- **Demographics:** Age, sex.
- **Label:** `convert_84m` (1 if the person converted to PD within ~84 months; else 0).

> Final training table: **1105** rows, with **~2.6% positives** (29/1105).

### 🧹 Preprocessing (kept simple & consistent)
- Drop obvious ID/leakage columns (e.g., `PATNO`).
- Median imputation for missing values.
- Standardization (scaling) for model input.
- **Class imbalance handled** via **SMOTE** inside the training pipeline.

### 🤖 Models we tried & what won
We compared several models on the same feature set:
- Logistic Regression, SVM (RBF), **Random Forest**, and XGBoost.
- Tuned them with randomized search + stratified 5-fold CV.
- **Random Forest performed best overall** for this dataset (strong ROC-AUC, stable calibration, simple to deploy).

**Current selection:** Random Forest (saved as `models/Best_Model_pipeline.pkl`), plus `models/scaler.pkl` and `models/feature_names.json`.

### ✅ Evaluation (no leakage)
To avoid any accidental leakage, we report **5-fold out-of-fold (OOF)** performance:
- **AUROC:** ~**0.981**
- **AUPRC:** ~**0.928**
- **Brier score:** ~**0.0073**
- **ECE (10-bin):** ~**0.029**
- Calibration plot written to `plots/calibration_curve.png`.
- Summary metrics written to `reports/quick_checks.json`.

*(Numbers above come from OOF evaluation, which better reflects generalization than scoring the full training set.)*

### 🖥️ Streamlit app (v1 features)
- **Predict (batch CSV):** Upload the same-schema CSV, get **probabilities + 0/1 predictions**, and download results.
- **Interpretability:** Optional per-patient **SHAP** bar chart (works if `shap` is installed and supported by the model).
- **Dashboard:** Shows latest metrics (AUROC/AUPRC, Brier, ECE) and the calibration curve.
- **Medical reminders:** Lightweight local tracker (SQLite) for follow-ups like “repeat UPSIT”.

### 📦 What gets saved (artifacts)
- `models/Best_Model_pipeline.pkl` – the full trained pipeline.
- `models/scaler.pkl` – the fitted scaler used by the app.
- `models/feature_names.json` – exact feature order/schema.
- `plots/calibration_curve.png` – reliability (calibration) curve.
- `reports/quick_checks.json` – headline evaluation metrics.

### 🧭 End-to-end flow (today)
1. Merge inputs → single CSV (`final_dataset_enhanced.csv`).
2. Train & tune models (handles imbalance with SMOTE).
3. Select **Random Forest** as best model; save artifacts.
4. Evaluate with OOF and write metrics/plots.
5. Use **Streamlit app** to upload data → get predictions + (optional) explanations → track reminders.

> Robust, interpretable baseline in place. Later phases can add deep models (RNN/LSTM) and a RAG-based clinical report layer without changing the current UI flow.


## 🚀 Quickstart

> **Prereqs:** Python 3.10+ and Git installed.

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

  

# 1) Clone
git clone https://github.com/gaya3jayan-11/Early-Prediction-of-PD.git <br/>
cd Early-Prediction-of-PD

# 2) Virtual env + deps
python -m venv .venv <br/>
. .\.venv\Scripts\Activate.ps1 <br/>
python -m pip install -U pip <br/>
pip install -r requirements.txt

# 3) Add data (skip if already in repo)
Expecting: data\processed\final_dataset_enhanced.csv

# 4) Train (saves artifacts to models/)
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8" <br/>
python -X utf8 -u scripts\model_training.py | Tee-Object reports\train_stdout.txt

# 5) Evaluate (OOF; writes reports/quick_checks.json, plots/calibration_curve.png)
python -X utf8 -u scripts\evaluate_oof.py

# 6) Run the app
streamlit run streamlit_app.py

# 7) Batch predict (writes CSVs in .\repo)
python scripts\predict.py -i data\processed\final_dataset_enhanced.csv -o repo

</details> <details> <summary><strong>macOS / Linux (Bash)</strong></summary>

# 1) Clone
git clone https://github.com/gaya3jayan-11/Early-Prediction-of-PD.git <br/>
cd Early-Prediction-of-PD

# 2) Virtual env + deps
python3 -m venv .venv <br/>
source .venv/bin/activate <br/>
python -m pip install -U pip <br/>
pip install -r requirements.txt

# 3) Add data (skip if already in repo)
Expecting: data/processed/final_dataset_enhanced.csv

# 4) Train (saves artifacts to models/)
python scripts/model_training.py | tee reports/train_stdout.txt

# 5) Evaluate (OOF; writes reports/quick_checks.json, plots/calibration_curve.png)
python scripts/evaluate_oof.py

# 6) Run the app
streamlit run streamlit_app.py

# 7) Batch predict (writes CSVs in ./repo)
python scripts/predict.py -i data/processed/final_dataset_enhanced.csv -o repo

</details>
