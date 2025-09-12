# scripts/predict.py
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import joblib

DEF_MODEL   = os.path.join("models", "Best_Model_pipeline.pkl")
DEF_SCALER  = os.path.join("models", "scaler.pkl")
DEF_FEATS   = os.path.join("models", "feature_names.json")

def load_artifacts(model_path, scaler_path, feats_path):
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"ERROR: scaler not found at {scaler_path}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(feats_path):
        print(f"ERROR: feature list not found at {feats_path}", file=sys.stderr); sys.exit(1)

    pipeline = joblib.load(model_path)     # imputer + sampler (ignored at predict) + model
    scaler   = joblib.load(scaler_path)    # StandardScaler fitted during training
    with open(feats_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    return pipeline, scaler, features

def prepare_matrix(df, features):
    # Drop obvious leakage/IDs if present (were removed during training)
    drop_if_present = ["PATNO", "convert_84m"]
    df = df.drop(columns=[c for c in drop_if_present if c in df.columns], errors="ignore")

    # Ensure all expected features exist; create missing as zeros (so scaler can run)
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} features missing in input CSV. "
              f"Filling with 0: {missing[:10]}{'...' if len(missing)>10 else ''}")
        for c in missing:
            df[c] = 0.0

    # Keep only the columns the model was trained on and order them
    X = df[features].copy()

    # Force numeric; unseen strings -> NaN -> fill 0 to keep scaler happy
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    return X

def main():
    ap = argparse.ArgumentParser(description="Predict PD conversion using saved best model.")
    ap.add_argument("-i", "--input",  required=True, help="Input CSV path")
    ap.add_argument("-o", "--output", default=os.path.join("reports","predictions.csv"), help="Output CSV path")
    ap.add_argument("--model",   default=DEF_MODEL,  help="Path to Best_Model_pipeline.pkl")
    ap.add_argument("--scaler",  default=DEF_SCALER, help="Path to scaler.pkl")
    ap.add_argument("--features",default=DEF_FEATS,  help="Path to feature_names.json")
    ap.add_argument("-t", "--threshold", type=float, default=0.50, help="Decision threshold (default 0.50)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading artifacts...")
    pipeline, scaler, features = load_artifacts(args.model, args.scaler, args.features)

    print(f"Reading: {args.input}")
    df_in = pd.read_csv(args.input, low_memory=False)
    id_col = "PATNO" if "PATNO" in df_in.columns else None

    X = prepare_matrix(df_in, features)

    print("Scaling...")
    Xs = scaler.transform(X)  # use the same scaler as training

    print("Predicting...")
    proba = pipeline.predict_proba(Xs)[:, 1]
    pred  = (proba >= args.threshold).astype(int)

    out = pd.DataFrame({
        **({"PATNO": df_in["PATNO"]} if id_col else {}),
        "prediction": pred,
        "probability": proba
    })

    # If ground-truth exists in the input, copy it through for quick checking
    if "convert_84m" in df_in.columns:
        out["convert_84m"] = df_in["convert_84m"]

    out.to_csv(args.output, index=False)
    print(f"Saved predictions -> {args.output}")
    print(f"Predicted positives at threshold {args.threshold:.2f}: {int(out['prediction'].sum())}/{len(out)}")

if __name__ == "__main__":
    main()
