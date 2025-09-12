from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import pandas as pd

def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path, nrows=nrows)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} for {path.name}")

def describe_basic(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "null_counts": df.isna().sum().to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head_preview": df.head(3).to_dict(orient="list"),
    }

def check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> Tuple[List[str], List[str]]:
    cols = set(df.columns)
    req = list(required)
    missing = [c for c in req if c not in cols]
    present = [c for c in req if c in cols]
    return present, missing

def coerce_dates(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def check_id_uniqueness(df: pd.DataFrame, id_col: str) -> dict:
    if id_col not in df.columns:
        return {"exists": False, "unique": None, "dupe_count": None}
    n = len(df)
    nunique = df[id_col].nunique(dropna=False)
    dupe_count = int((n - nunique))
    return {"exists": True, "unique": (nunique == n), "dupe_count": dupe_count}

def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return miss.to_frame(name="missing_rate").reset_index(names="column")
