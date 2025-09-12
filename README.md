## 🚀 Quickstart

> **Prereqs:** Python 3.10+ and Git installed.

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

  

# 1) Clone
git clone https://github.com/<you>/pd-risk-prediction.git
cd pd-risk-prediction

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
git clone https://github.com/<you>/pd-risk-prediction.git <br/>
cd pd-risk-prediction

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
