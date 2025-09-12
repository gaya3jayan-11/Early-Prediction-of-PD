## 🚀 Quickstart

> **Prereqs:** Python 3.10+ and Git installed.

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

# 1) Clone
git clone https://github.com/<you>/pd-risk-prediction.git
cd pd-risk-prediction

# 2) Virtual env + deps
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

# 3) Add data
# Place your dataset at:
# data\processed\final_dataset_enhanced.csv

# 4) Train (saves artifacts to models/)
# (UTF-8 flags avoid Windows encoding glitches)
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
python -X utf8 -u scripts\model_training.py | Tee-Object reports\train_stdout.txt

# 5) Evaluate (OOF; creates reports\quick_checks.json and plots\calibration_curve.png)
python -X utf8 -u scripts\evaluate_oof.py

# 6) Run the app
streamlit run streamlit_app.py

# 7) Batch predict (writes CSVs in .\repo)
python scripts\predict.py -i data\processed\final_dataset_enhanced.csv -o repo


</details> <details> <summary><strong>macOS / Linux (Bash)</strong></summary>

# 1) Clone
git clone https://github.com/<you>/pd-risk-prediction.git
cd pd-risk-prediction

# 2) Virtual env + deps
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# 3) Add data
# Place your dataset at:
# data/processed/final_dataset_enhanced.csv

# 4) Train (saves artifacts to models/)
python scripts/model_training.py | tee reports/train_stdout.txt

# 5) Evaluate (OOF; creates reports/quick_checks.json and plots/calibration_curve.png)
python scripts/evaluate_oof.py

# 6) Run the app
streamlit run streamlit_app.py

# 7) Batch predict (writes CSVs in ./repo)
python scripts/predict.py -i data/processed/final_dataset_enhanced.csv -o repo

</details>
