from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import re

PROJ = Path(__file__).resolve().parents[1]
CFG = PROJ / "configs" / "data.yaml"
PROC = PROJ / "data" / "processed"
REPS = PROJ / "reports"

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def smart_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan

def main():
    cfg = load_yaml(CFG)
    id_col = cfg["id_column"]
    ycol = cfg.get("outcome_column", "convert_84m")

    src = PROC / "master_baseline.parquet"
    assert src.exists(), f"Missing {src}"
    df = pd.read_parquet(src)

    # --- base columns ---
    must_have = [id_col, ycol, "baseline_date_final"]
    for c in must_have:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    # Demographics
    demo_cols = []
    if "age_at_baseline" in df: demo_cols.append("age_at_baseline")
    if "sex_norm" in df:
        df["sex_norm_num"] = df["sex_norm"].map({"M": 1.0, "F": 0.0})
        demo_cols.append("sex_norm_num")

    # Plasma features
    plasma_cols = [c for c in df.columns if c.startswith("plasma__")]

    # Drop obvious metadata-like plasma cols
    drop_like = ("patno", "project", "pi_", "cohort", "clinical_event",
                 "type", "status", "update", "run", "institution", "name")
    plasma_cols = [c for c in plasma_cols if not any(x in c.lower() for x in drop_like)]

    # Coerce any non-numeric plasma to numeric if possible
    for c in plasma_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(smart_to_float)

    # Remove constant columns
    plasma_cols = [c for c in plasma_cols if df[c].nunique(dropna=True) > 1]

    # Rank plasma features by coverage
    cov = pd.Series({c: df[c].notna().mean() for c in plasma_cols}).sort_values(ascending=False)

    # Auto-selection with fallback:
    
    keep_plasma = cov[cov >= 0.05].index.tolist()
    if not keep_plasma:
        keep_plasma = cov.head(150).index.tolist()

    feature_cols = demo_cols + keep_plasma

    # Row filter: require baseline, label, and at least K non-null features
    def build_rows(min_nonnull_feats: int):
        # columns to use
        use = [id_col, ycol, "baseline_date_final"] + feature_cols
        use = [c for c in use if c in df.columns]
        Xy = df[use].copy()
        # always require label + baseline date
        base_mask = Xy["baseline_date_final"].notna() & Xy[ycol].notna()
        # keep rows with >= K non-null features …
        feat_ok = (Xy[feature_cols].notna().sum(axis=1) >= min_nonnull_feats)
        # … OR any positive label (force-keep positives)
        pos_mask = (Xy[ycol] == 1)

        row_mask = base_mask & (feat_ok | pos_mask)
        return Xy.loc[row_mask, [id_col, ycol] + feature_cols].copy()

    Xy = build_rows(min_nonnull_feats=3)
    if len(Xy) < 800:  # relax progressively
        Xy = build_rows(min_nonnull_feats=2)
    if len(Xy) < 400:
        Xy = build_rows(min_nonnull_feats=1)

    # If still empty, fallback to demographics-only (age/sex)
    if len(Xy) == 0 and demo_cols:
        feature_cols = demo_cols
        Xy = df[[id_col, ycol, "baseline_date_final"] + feature_cols].dropna(subset=[ycol, "baseline_date_final"], how="any")
        # allow 0 non-null feature rows to pass; we’ll impute

    # If STILL empty, bail with a clear error
    if len(Xy) == 0:
        raise RuntimeError("After relaxed curation there are still 0 rows. We need more baseline dates or looser alignment upstream.")

    # Final impute medians for numeric features
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(Xy[c]):
            Xy[c] = pd.to_numeric(Xy[c], errors="coerce")
        med = Xy[c].median(skipna=True)
        Xy[c] = Xy[c].fillna(med)

    # Save
    PROC.mkdir(parents=True, exist_ok=True)
    REPS.mkdir(parents=True, exist_ok=True)

    out_parquet = PROC / "training_matrix.parquet"
    out_csv = PROC / "training_matrix_preview.csv"
    out_cov = REPS / "feature_curation_report.csv"

    # Export coverage report
    cov_df = pd.DataFrame({"feature": list(cov.index), "nonnull_frac": cov.values})
    cov_df["kept"] = cov_df["feature"].isin(keep_plasma)

    Xy.to_parquet(out_parquet, index=False)
    Xy.head(200).to_csv(out_csv, index=False)
    cov_df.to_csv(out_cov, index=False)

    # Print summary
    pos = int(Xy[ycol].sum())
    pos_rate = float(Xy[ycol].mean()) if len(Xy) else 0.0
    print(f"✅ training_matrix: {out_parquet}")
    print(f"👀 preview: {out_csv}")
    print(f"📊 curation report: {out_cov}")
    print(f"rows kept: {len(Xy):,} | features kept: {len(feature_cols):,} | positives: {pos:,} ({pos_rate:.2%})")

if __name__ == "__main__":
    main()
