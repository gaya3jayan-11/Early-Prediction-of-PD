from pathlib import Path
import json, numpy as np, pandas as pd, sys, traceback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler  # stable 1:3 oversampling

PROJ = Path(__file__).resolve().parents[1]
PROC = PROJ / "data" / "processed"
REPS = PROJ / "reports"
MODELS = PROJ / "models"

def best_threshold_pr(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i]), float(p[i]), float(r[i])

def main():
    print(">> starting trainer", flush=True)
    REPS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    p_tm = PROC / "training_matrix.parquet"
    print(f">> loading: {p_tm}", flush=True)
    df = pd.read_parquet(p_tm)
    ycol = "convert_84m" if "convert_84m" in df.columns else [c for c in df.columns if str(c).startswith("convert_")][0]
    idc = "subject_id" if "subject_id" in df.columns else None

    drop_cols = [c for c in [idc, "baseline_date_final", ycol] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[ycol].astype(int)
    print(f">> rows: {len(df)} | features: {X.shape[1]} | positives: {int(y.sum())}", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pos_tr = int(ytr.sum())
    print(f">> train size: {len(ytr)} (pos={pos_tr}) | test size: {len(yte)} (pos={int(yte.sum())})", flush=True)
    assert pos_tr >= 10, f"Too few positives in train split: {pos_tr}"

    pipe = ImbPipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("ros", RandomOverSampler(sampling_strategy=0.15, random_state=42)),  # ~1:3
        ("scale", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])

    print(">> fitting...", flush=True)
    pipe.fit(Xtr, ytr)
    print(">> predicting...", flush=True)
    y_prob = pipe.predict_proba(Xte)[:, 1]

    roc = float(roc_auc_score(yte, y_prob))
    auprc = float(average_precision_score(yte, y_prob))
    brier = float(brier_score_loss(yte, y_prob))
    thr, f1, prec, rec = best_threshold_pr(yte.values, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(yte, y_pred).tolist()

    # save artifacts
    import joblib
    model_path = MODELS / "logreg_ros_1to3.joblib"
    preds_path = REPS / "resampled_predictions.csv"
    metrics_path = REPS / "resampled_metrics.json"

    print(f">> saving model: {model_path}", flush=True)
    joblib.dump({"model": pipe, "threshold": thr}, model_path)

    print(f">> saving predictions: {preds_path}", flush=True)
    pd.DataFrame({
        (idc if idc else "row_id"): (df.loc[yte.index, idc] if idc else yte.index),
        "y_true": yte.values, "y_prob": y_prob, "y_pred@thr": y_pred
    }).to_csv(preds_path, index=False)

    metrics = {
        "roc_auc": roc, "auprc": auprc, "brier": brier,
        "optimal_threshold_f1": thr, "f1_at_thr": f1,
        "precision_at_thr": prec, "recall_at_thr": rec,
        "confusion_matrix_at_thr": cm,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "positives_train": int(ytr.sum()), "positives_test": int(yte.sum()),
        "sampling_strategy": 0.33,
        "model_path": str(model_path), "preds_path": str(preds_path)
    }
    print(f">> saving metrics: {metrics_path}", flush=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("done", flush=True)
    print(metrics, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        REPS.mkdir(parents=True, exist_ok=True)
        err_path = REPS / "resampled_error.txt"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("Error. See", err_path, flush=True)
        sys.exit(1)