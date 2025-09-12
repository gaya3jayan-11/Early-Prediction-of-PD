from pathlib import Path
import json, numpy as np, pandas as pd, sys, traceback
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, confusion_matrix

PROJ = Path(__file__).resolve().parents[1]
PROC = PROJ / "data" / "processed"
REPS = PROJ / "reports"
MODELS = PROJ / "models"

def best_threshold_pr(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i]), float(p[i]), float(r[i])

def main():
    print(">> HGB trainer starting", flush=True)
    REPS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC / "training_matrix.parquet")
    ycol = "convert_84m" if "convert_84m" in df.columns else [c for c in df.columns if str(c).startswith("convert_")][0]
    idc = "subject_id" if "subject_id" in df.columns else None

    drop_cols = [c for c in [idc, "baseline_date_final", ycol] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[ycol].astype(int)

    # Drop all-NaN or constant columns (robustness)
    to_drop = [c for c in X.columns if X[c].notna().sum()==0 or X[c].nunique(dropna=True)<=1]
    if to_drop:
        print(f">> dropping {len(to_drop)} degenerate cols: {to_drop[:10]}...", flush=True)
        X = X.drop(columns=to_drop)

    # Impute (safer with mixed dtypes)
    X = SimpleImputer(strategy="median").fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Class weighting via sample_weight
    pos_w = (len(ytr) - int(ytr.sum())) / max(1, int(ytr.sum()))
    w = np.where(ytr == 1, pos_w, 1.0)

    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=400, random_state=42)
    clf.fit(Xtr, ytr, sample_weight=w)
    y_prob = clf.predict_proba(Xte)[:, 1]

    roc = float(roc_auc_score(yte, y_prob))
    auprc = float(average_precision_score(yte, y_prob))
    brier = float(brier_score_loss(yte, y_prob))
    thr, f1, prec, rec = best_threshold_pr(yte.values, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(yte, y_pred).tolist()

    # Save
    import joblib
    model_path = MODELS / "hgb_weighted.joblib"
    preds_path = REPS / "hgb_predictions.csv"
    metrics_path = REPS / "hgb_metrics.json"
    joblib.dump({"model": clf, "threshold": thr}, model_path)

    pd.DataFrame({"y_true": yte.values, "y_prob": y_prob, "y_pred@thr": y_pred}).to_csv(preds_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "roc_auc": roc, "auprc": auprc, "brier": brier,
            "optimal_threshold_f1": thr, "f1_at_thr": f1, "precision_at_thr": prec, "recall_at_thr": rec,
            "confusion_matrix_at_thr": cm,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "model_path": str(model_path), "preds_path": str(preds_path)
        }, f, indent=2)
    print(">> HGB done. Metrics -> reports\\hgb_metrics.json", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        REPS.mkdir(parents=True, exist_ok=True)
        with open(REPS / "hgb_error.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("ERROR. See reports\\hgb_error.txt", flush=True)
        sys.exit(1)