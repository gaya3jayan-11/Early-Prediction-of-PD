from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve

PROJ = Path(__file__).resolve().parents[1]
PROC = PROJ / "data" / "processed"
REPS = PROJ / "reports"

def pr_best_f1(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    f1 = 2*p*r/(p+r+1e-12)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i]), float(p[i]), float(r[i])

def main():
    df = pd.read_parquet(PROC / "training_matrix.parquet")
    ycol = "convert_84m" if "convert_84m" in df.columns else [c for c in df.columns if str(c).startswith("convert_")][0]
    idc = "subject_id" if "subject_id" in df.columns else None

    drop_cols = [c for c in [idc, "baseline_date_final", ycol] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    y = df[ycol].astype(int).to_numpy()

    # 5-fold stratified CV, manual loop to keep sample weights + proper fit/transform per fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y))
    oof_idx = np.zeros(len(y), dtype=bool)
    fold_metrics = []

    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        steps = []
        steps.append(("impute", SimpleImputer(strategy="median")))
        steps.append(("var", VarianceThreshold(threshold=0.0))) # drop constants based on TRAIN fold
        # HGB with sample weighting
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=400, random_state=42)
        # fit imputer+var on TRAIN; transform TRAIN & VAL
        Xt = steps[0][1].fit_transform(Xtr)
        Xt = steps[1][1].fit_transform(Xt)
        Xv = steps[0][1].transform(Xva)
        Xv = steps[1][1].transform(Xv)

        pos_w = (len(ytr) - int(ytr.sum())) / max(1, int(ytr.sum()))
        w = np.where(ytr == 1, pos_w, 1.0)
        clf.fit(Xt, ytr, sample_weight=w)
        prob = clf.predict_proba(Xv)[:, 1]

        # fold metrics
        roc = float(roc_auc_score(yva, prob))
        auprc = float(average_precision_score(yva, prob))
        brier = float(brier_score_loss(yva, prob))
        thr, f1, prec, rec = pr_best_f1(yva, prob)
        fold_metrics.append({"fold": k, "roc_auc": roc, "auprc": auprc, "brier": brier,
                             "thr": thr, "f1": f1, "precision": prec, "recall": rec,
                             "n_val": int(len(yva)), "pos_val": int(yva.sum())})

        oof_prob[va] = prob
        oof_idx[va] = True

    # overall OOF metrics
    assert oof_idx.all(), "OOF coverage incomplete"
    roc = float(roc_auc_score(y, oof_prob))
    auprc = float(average_precision_score(y, oof_prob))
    brier = float(brier_score_loss(y, oof_prob))
    thr, f1, prec, rec = pr_best_f1(y, oof_prob)

    REPS.mkdir(parents=True, exist_ok=True)
    (REPS / "hgb_cv_oof.csv").write_text(
        pd.DataFrame({"y_true": y, "y_prob": oof_prob}).to_csv(index=False),
        encoding="utf-8"
    )
    with open(REPS / "hgb_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"folds": fold_metrics,
                   "oof": {"roc_auc": roc, "auprc": auprc, "brier": brier,
                           "optimal_threshold_f1": thr, "f1_at_thr": f1,
                           "precision_at_thr": prec, "recall_at_thr": rec}},
                  f, indent=2)
    print("Saved reports\\hgb_cv_metrics.json and reports\\hgb_cv_oof.csv")

if __name__ == "__main__":
    main()