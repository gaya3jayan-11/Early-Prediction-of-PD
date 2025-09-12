from __future__ import annotations

from src.data.load_ppmi import ensure_unique_columns, drop_duplicate_id_columns
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np

from src.data.load_ppmi import (
    read_table, load_yaml, coerce_dates, normalize_id,
    pick_first_existing, get_baseline_dates, derive_conversion_label,
    build_plasma_features
)

def load_files(cfg: Dict[str, Any], raw_root: Path) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
    from src.data.load_ppmi import ensure_unique_columns, drop_duplicate_id_columns

    def _clean(df: pd.DataFrame, raw_id: str, out_id: str) -> pd.DataFrame:
        df = df.copy()
        df = ensure_unique_columns(df)
        df = df.loc[:, ~df.columns.duplicated()]
        # normalize id, then remove any extra copies of out_id
        df = normalize_id(df, raw_id, out_id)
        df = drop_duplicate_id_columns(df, out_id)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    raw_id = cfg["join_key_raw"]
    out_id = cfg["id_column"]

    # Demographics
    demop = raw_root / cfg["files"]["demographics"]
    demo = _clean(read_table(demop), raw_id, out_id)

    # Diagnosis events
    dxp = raw_root / cfg["files"]["diagnosis_events"]
    dx = _clean(read_table(dxp), raw_id, out_id)

    # Plasma (allow list or single)
    plasma_cfg = cfg["files"].get("plasma_biomarkers", [])
    if isinstance(plasma_cfg, str):
        plasma_cfg = [plasma_cfg]
    plasma_tables = []
    for rel in plasma_cfg:
        pp = raw_root / rel
        t = _clean(read_table(pp), raw_id, out_id)
        plasma_tables.append(t)

    # Optional instruments (list)
    inst_cfg = cfg["files"].get("instruments_optional", [])
    instruments = []
    for rel in inst_cfg:
        pp = raw_root / rel
        if pp.exists():
            t = _clean(read_table(pp), raw_id, out_id)
            instruments.append(t)

    return {
        "demographics": demo,
        "diagnosis_events": dx,
        "plasma_tables": plasma_tables,
        "instruments": instruments
    }



def enrich_demographics(demo: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = demo.copy()
    # Normalize sex and birthdate if present
    sex_col = pick_first_existing(out, ["SEX", "GENDER"])
    bdt_col = pick_first_existing(out, ["BIRTHDT", "BIRTH_DATE"])
    if bdt_col:
        out = coerce_dates(out, [bdt_col])
    if sex_col:
        out["sex_norm"] = (out[sex_col].astype(str)
                           .str.strip()
                           .str.upper()
                           .map({"M": "M", "MALE":"M", "F":"F", "FEMALE":"F"})
                           .fillna("UNK"))
    else:
        out["sex_norm"] = "UNK"
    if bdt_col:
        out["birth_date_norm"] = out[bdt_col]
    else:
        out["birth_date_norm"] = pd.NaT
    return out

def compute_age_at_baseline(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    if "birth_date_norm" in out.columns and "baseline_date_final" in out.columns:
        out["age_at_baseline"] = (
            (out["baseline_date_final"] - out["birth_date_norm"]).dt.days / 365.25
        )
    return out

def assemble_master(cfg: Dict[str, Any], paths: Dict[str, str]) -> pd.DataFrame:
    raw_root = Path(paths["raw_dir"])
    id_col = cfg["id_column"]
    outcome_name = cfg.get("outcome_column", "convert_24m")          # <-- new
    event_cands = ["EVENT_ID", "EVENTID", "VISIT", "VISCOD", "VISIT_CODE"]
    exam_cands = cfg["exam_date_candidates"]
    dx_cands = cfg["diagnosis_date_candidates"]
    baseline_code = cfg["baseline_event_code"]

    # Load & enrich
    loaded = load_files(cfg, raw_root)
    demo = enrich_demographics(loaded["demographics"], cfg)

    # Baseline date extraction
    base = pd.DataFrame({id_col: demo[id_col].unique()})
    if loaded["instruments"]:
        bl = get_baseline_dates(
            instruments=loaded["instruments"],
            id_col=id_col,
            event_col_candidates=event_cands,
            exam_date_candidates=exam_cands,
            baseline_event_code=baseline_code
        )
        base = base.merge(bl, on=id_col, how="left")
    else:
        base["baseline_date_final"] = pd.NaT
        base["baseline_source"] = "none"

    # Labels from diagnosis history (uses dynamic outcome name and window)
    lab = derive_conversion_label(
        dx=loaded["diagnosis_events"],
        id_col=id_col,
        diagnosis_date_candidates=dx_cands,
        baseline=base[[id_col, "baseline_date_final"]],
        window_months=cfg["window_months"],
        outcome_name=outcome_name,                                    # <-- new
    )

    # Plasma features near baseline
    plasma = build_plasma_features(
        plasma_tables=loaded["plasma_tables"],
        id_col=id_col,
        exam_date_candidates=exam_cands,
        baseline=lab[[id_col, "baseline_date_final"]],
        nearest_days=730
    )

    # Join everything
    master = (demo[[id_col, "sex_norm", "birth_date_norm"]]
              .merge(lab, on=id_col, how="left")
              .merge(plasma, on=id_col, how="left"))

    master = compute_age_at_baseline(master, id_col)

    # Fill missing label with 0 if baseline is present but no dx found
    if outcome_name in master.columns:
        mask_baseline_known = master["baseline_date_final"].notna()
        master.loc[mask_baseline_known & master[outcome_name].isna(), outcome_name] = 0

    # Final column order
    front = [id_col, "baseline_date_final", outcome_name, "age_at_baseline", "sex_norm"]
    others = [c for c in master.columns if c not in front]
    master = master[front + others]

    plasma_meta = [
        "plasma__PATNO", "plasma__SEX", "plasma__COHORT", "plasma__CLINICAL_EVENT",
        "plasma__TYPE", "plasma__PROJECTID", "plasma__PI_NAME", "plasma__PI_INSTITUTION",
        "plasma__update_stamp", "plasma__RUNDATE"
    ]
    master = master.drop(columns=[c for c in plasma_meta if c in master.columns], errors="ignore")

    # Make sure there is only one id column, and all column names are unique
    master = drop_duplicate_id_columns(master, id_col)
    master = ensure_unique_columns(master)
    master = master.loc[:, ~master.columns.duplicated()]

    return master
