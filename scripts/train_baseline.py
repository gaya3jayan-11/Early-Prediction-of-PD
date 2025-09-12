
from pathlib import Path
import json, yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, confusion_matrix
)

PROJ = Path(__file__).resolve().parents[1]
PROC = PROJ / "data" / "processed"
REPS = PROJ / "reports"
MODELS = PROJ / "models"
CFG = PROJ / "configs" / "model.yaml"

def load_yaml(p):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def main():
    REPS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(CFG)
    target = cfg.get("target", "convert_84m")
    rs = int(cfg.get("random_state", 42))

    df = pd.read_parquet(PROC / "training_matrix.parquet")

    id_col = "subject_id" if "subject_id" in df.columns else None
    assert target in df.columns, f"Missing target: {target}"

    drop_cols = [c for c in [id_col, "baseline_date_final", target] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target].astype(int)

    pos = int(y.sum())
    assert pos >= 10, f"Not enough positives to train (found {pos})."

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)

    base = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    # use 'estimator' (new API); 'sigmoid' is stabler with few positives
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    pipe.fit(Xtr, ytr)
    y_prob = pipe.predict_proba(Xte)[:, 1]

    # metrics
    roc = float(roc_auc_score(yte, y_prob))
    auprc = float(average_precision_score(yte, y_prob))
    brier = float(brier_score_loss(yte, y_prob))

    # threshold tuning (maximize F1)
    p, r, t = precision_recall_curve(yte, y_prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    opt_thr = float(np.append(t, 1.0)[i])  # align sizes

    y_pred = (y_prob >= opt_thr).astype(int)
    cm = confusion_matrix(yte, y_pred).tolist()

    # save artifacts
    import joblib
    model_path = MODELS / "baseline_logreg_calibrated.joblib"
    joblib.dump({"model": pipe, "threshold": opt_thr}, model_path)

    preds = pd.DataFrame({
        (id_col if id_col else "row_id"): (df.loc[yte.index, id_col] if id_col else yte.index),
        "y_true": yte.values,
        "y_prob": y_prob,
        "y_pred@thr": y_pred
    })
    preds_path = REPS / "baseline_predictions.csv"
    preds.to_csv(preds_path, index=False)

    metrics = {
        "roc_auc": roc,
        "auprc": auprc,
        "brier": brier,
        "optimal_threshold_f1": opt_thr,
        "confusion_matrix_at_thr": cm,
        "positives_test": int(yte.sum()),
        "n_test": int(len(yte)),
        "n_train": int(len(ytr)),
        "n_features": int(X.shape[1])
    }
    metrics_path = REPS / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Trained baseline model (imputer + varfilter + sigmoid-calibrated)")
    print(metrics)

if __name__ == "__main__":
    main()