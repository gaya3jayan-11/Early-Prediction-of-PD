from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import yaml
import re

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(map(str, df.columns))
    seen = {}
    new = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new.append(f"{c}__dup{seen[c]}")
        else:
            seen[c] = 0
            new.append(c)
    out = df.copy()
    out.columns = new
    return out

def smart_to_float(x):
    """Parse ' <0.1 ', '0,85', 'ND' -> 0.1 / 0.85 / NaN."""
    if pd.isna(x): return pd.NA
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else pd.NA

def drop_duplicate_id_columns(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    # keep the first id_col, drop any extra same-named columns
    mask = [c == id_col for c in df.columns]
    if sum(mask) > 1:
        # drop all but the first
        to_drop = [i for i, m in enumerate(mask) if m][1:]
        out = df.drop(df.columns[to_drop], axis=1)
        return out
    return df


# ---------- basic readers ----------
def read_table(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False, encoding_errors="ignore")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {p.suffix} for {p.name}")

def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def coerce_dates(df: pd.DataFrame, candidates: Iterable[str]) -> pd.DataFrame:
    # Try common formats first to reduce warnings; fallback to dateutil
    for c in candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="raise")
            except Exception:
                try:
                    df[c] = pd.to_datetime(df[c], format="%d-%b-%Y", errors="raise")
                except Exception:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def normalize_id(df: pd.DataFrame, raw_id: str, out_id: str) -> pd.DataFrame:
    if raw_id not in df.columns:
        raise KeyError(f"ID column {raw_id} not in columns: {df.columns.tolist()[:20]}...")
    out = df.copy()
    out[out_id] = out[raw_id]
    return out

# ---------- domain helpers ----------
def pick_first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def get_baseline_dates(
    instruments: List[pd.DataFrame],
    id_col: str,
    event_col_candidates: Iterable[str],
    exam_date_candidates: Iterable[str],
    baseline_event_code: str,
) -> pd.DataFrame:
    """
    Find baseline (BL) date per subject using any instrument that has EVENT_ID + an exam/visit date.
    Also collect the earliest exam date even if EVENT_ID isn't present (robust fallback).
    Returns: [id_col, baseline_date_final, baseline_source]
    """
    rows = []
    for t in instruments:
        t = t.copy()

        # Identify date column (required for any signal)
        ex_col = pick_first_existing(t, exam_date_candidates)
        if ex_col is None:
            continue
        t = coerce_dates(t, [ex_col])

        # If an event column exists, try explicit BL
        ev_col = pick_first_existing(t, event_col_candidates)
        if ev_col is not None and ev_col in t.columns:
            bl = t.loc[t[ev_col] == baseline_event_code, [id_col, ex_col]].copy()
            if not bl.empty:
                bl = (bl.groupby(id_col, as_index=False)[ex_col]
                        .min()
                        .rename(columns={ex_col: "baseline_date"}))
                bl["source"] = "explicit_BL"
                rows.append(bl)

        # Always compute earliest exam date per subject as fallback
        anyd = (t.groupby(id_col, as_index=False)[ex_col]
                  .min()
                  .rename(columns={ex_col: "earliest_date"}))
        anyd["source"] = "earliest_any"
        rows.append(anyd)

    # If nothing usable, return empty frame with expected columns
    if not rows:
        return pd.DataFrame(columns=[id_col, "baseline_date_final", "baseline_source"])

    merged = pd.concat(rows, axis=0, ignore_index=True)

    # Pivot to combine explicit BL and earliest_any
    # After concat we have either 'baseline_date' or 'earliest_date' as value column with 'source' label.
    # Normalize to two columns via pivot.
    # First, melt to a common 'value' column
    value_col = "value"
    m = []
    for g, sub in merged.groupby("source", as_index=False):
        if "baseline_date" in sub.columns:
            x = sub[[id_col, "baseline_date"]].copy()
            x[value_col] = x["baseline_date"]
            x["kind"] = "baseline_date"
            m.append(x[[id_col, value_col, "kind"]])
        if "earliest_date" in sub.columns:
            x = sub[[id_col, "earliest_date"]].copy()
            x[value_col] = x["earliest_date"]
            x["kind"] = "earliest_date"
            m.append(x[[id_col, value_col, "kind"]])
    if not m:
        return pd.DataFrame(columns=[id_col, "baseline_date_final", "baseline_source"])

    melted = pd.concat(m, ignore_index=True)
    # Reduce: earliest across duplicates for each kind
    pivot = (melted
             .dropna(subset=[value_col])
             .sort_values([id_col, value_col])
             .drop_duplicates([id_col, "kind"]))
    # Prepare final columns
    bl_map = pivot[pivot["kind"] == "baseline_date"].set_index(id_col)[value_col]
    ea_map = pivot[pivot["kind"] == "earliest_date"].set_index(id_col)[value_col]

    idx = sorted(set(bl_map.index).union(ea_map.index))
    out = pd.DataFrame({id_col: idx})
    out["baseline_date_final"] = out[id_col].map(bl_map)
    out["baseline_source"] = np.where(out["baseline_date_final"].notna(), "explicit_BL", "earliest_any")
    out.loc[out["baseline_date_final"].isna(), "baseline_date_final"] = out.loc[
        out["baseline_date_final"].isna(), id_col
    ].map(ea_map)

    return out[[id_col, "baseline_date_final", "baseline_source"]]

def derive_conversion_label(
    dx: pd.DataFrame,
    id_col: str,
    diagnosis_date_candidates: Iterable[str],
    baseline: pd.DataFrame,
    window_months: int,
    outcome_name: str = "convert_24m",   # <-- new param
) -> pd.DataFrame:
    """
    Label = 1 if earliest diagnosis date occurs within 'window_months' after baseline date, and not before baseline.
    """
    dx = dx.copy()
    dcol = pick_first_existing(dx, diagnosis_date_candidates)
    if dcol is None:
        # try EXAMDATE as fallback
        dcol = pick_first_existing(dx, ["EXAMDATE", "EXAM_DATE", "EVENT_DATE"])
    if dcol is None:
        # If no usable date at all, return ids-only negatives
        out = baseline[[id_col]].copy()
        out[outcome_name] = 0
        out["baseline_date_final"] = baseline["baseline_date_final"]
        return out[[id_col, "baseline_date_final", outcome_name]]

    dx = coerce_dates(dx, [dcol])

    # earliest diagnosis per subject
    dmin = dx.groupby(id_col, as_index=False)[dcol].min().rename(columns={dcol: "first_dx_date"})

    lab = baseline.merge(dmin, on=id_col, how="left")
    lab[outcome_name] = 0
    lab["baseline_date_final"] = pd.to_datetime(lab["baseline_date_final"], errors="coerce")
    lab["first_dx_date"] = pd.to_datetime(lab["first_dx_date"], errors="coerce")

    # condition: diagnosis exists and within window, and not before baseline
    mask_valid = lab["first_dx_date"].notna() & lab["baseline_date_final"].notna()
    lab.loc[
        mask_valid
        & (lab["first_dx_date"] >= lab["baseline_date_final"])
        & (lab["first_dx_date"] <= lab["baseline_date_final"] + pd.DateOffset(months=window_months)),
        outcome_name,
    ] = 1

    return lab[[id_col, "baseline_date_final", outcome_name]]


def _coerce_numeric_inplace(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    ex = set(exclude)
    for c in df.columns:
        if c in ex:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        # attempt conversion
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    return df

def build_plasma_features(
    plasma_tables: List[pd.DataFrame],
    id_col: str,
    exam_date_candidates: Iterable[str],
    baseline: pd.DataFrame,
    nearest_days: int = 730,
) -> pd.DataFrame:
    """
    Robust plasma builder:
      • If table has TESTNAME/TESTVALUE -> pivot TESTNAME into columns.
      • Parse numeric strings in TESTVALUE (e.g., '<0.1').
      • Align to baseline within ±nearest_days if possible; if coverage stays low, fall back to
        'latest by date' per [id, TESTNAME]; otherwise plain mean per [id, TESTNAME].
      • For non-analyte tables (e.g., SAA with replicate cols), coerce numerics and average per subject,
        after optional nearest-date selection.
    """
    feats = []
    for t in plasma_tables:
        t = ensure_unique_columns(t.copy())
        t = drop_duplicate_id_columns(t, id_col)
        t = t.loc[:, ~t.columns.duplicated()]

        dcol = pick_first_existing(t, exam_date_candidates)
        if dcol is not None:
            t = coerce_dates(t, [dcol])

        # -------- Case A: generic analyte table (TESTNAME + TESTVALUE)
        if {"TESTNAME", "TESTVALUE"}.issubset(t.columns):
            tmp = t[[c for c in [id_col, dcol, "TESTNAME", "TESTVALUE"] if c in t.columns]].copy()
            tmp["TESTVALUE"] = tmp["TESTVALUE"].map(smart_to_float)

            def pivot_per(tmp_):
                agg = tmp_.groupby([id_col, "TESTNAME"], as_index=False)["TESTVALUE"].mean()
                wide = agg.pivot(index=id_col, columns="TESTNAME", values="TESTVALUE").reset_index()
                wide.columns = [id_col if c == id_col else f"plasma__{str(c).strip().lower().replace(' ', '_')}" for c in wide.columns]
                return wide

            # Path 1: try baseline alignment if we have dates
            aligned = None
            coverage = 0.0
            if dcol and dcol in tmp.columns and "baseline_date_final" in baseline.columns:
                tmp1 = tmp.merge(baseline[[id_col, "baseline_date_final"]], on=id_col, how="inner")
                tmp1["delta_days"] = (tmp1[dcol] - tmp1["baseline_date_final"]).dt.days.abs()
                if tmp1["delta_days"].notna().any():
                    within = tmp1[tmp1["delta_days"] <= nearest_days]
                    if within.empty:
                        within = tmp1.sort_values([id_col, "TESTNAME", "delta_days"]).groupby([id_col,"TESTNAME"], as_index=False).first()
                    else:
                        within = within.sort_values([id_col, "TESTNAME", "delta_days"]).groupby([id_col,"TESTNAME"], as_index=False).first()
                    aligned = pivot_per(within.drop(columns=[c for c in ["delta_days", dcol, "baseline_date_final"] if c in within.columns]))
                    # rough coverage: subjects with any non-null value
                    if aligned.shape[0] > 0:
                        nonnull_rows = (aligned.drop(columns=[id_col], errors="ignore").notna().sum(axis=1) > 0).sum()
                        coverage = nonnull_rows / max(aligned.shape[0], 1)

            # Path 2: fallback if alignment coverage too low -> pick latest by date, else plain mean
            if aligned is None or coverage < 0.25:
                if dcol and dcol in tmp.columns:
                    tmp2 = tmp.sort_values([id_col, "TESTNAME", dcol]).groupby([id_col,"TESTNAME"], as_index=False).tail(1)
                    fallback = pivot_per(tmp2)
                else:
                    fallback = pivot_per(tmp)
                feats.append(fallback)
            else:
                feats.append(aligned)
            continue

        # -------- Case B: non-analyte table (e.g., SAA replicates)
        t = _coerce_numeric_inplace(t, exclude=[id_col] + ([dcol] if dcol else []))
        keep = [id_col] + ([dcol] if dcol else []) + [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c])]
        t = t[[c for c in keep if c in t.columns]]
        t = ensure_unique_columns(t)
        t = drop_duplicate_id_columns(t, id_col)
        t = t.loc[:, ~t.columns.duplicated()]

        if dcol and dcol in t.columns and "baseline_date_final" in baseline.columns:
            tmp = t.merge(baseline[[id_col, "baseline_date_final"]], on=id_col, how="inner")
            tmp["delta_days"] = (tmp[dcol] - tmp["baseline_date_final"]).dt.days.abs()
            within = tmp[tmp["delta_days"] <= nearest_days] if tmp["delta_days"].notna().any() else tmp
            if within.empty:
                within = tmp
            # nearest row per subject
            t = (within.sort_values([id_col, "delta_days"]).groupby(id_col, as_index=False).first())
            t = t.drop(columns=[c for c in ["delta_days", dcol, "baseline_date_final"] if c in t.columns])
        else:
            # average per subject (numeric only)
            t = t.groupby(id_col, as_index=False).mean(numeric_only=True)

        # prefix
        t = t.rename(columns={c: (c if c == id_col else f"plasma__{str(c)}") for c in t.columns})
        feats.append(t)

    if not feats:
        return pd.DataFrame(columns=[id_col])

    out = feats[0]
    for f in feats[1:]:
        out = out.merge(f, on=id_col, how="outer")
    return out
