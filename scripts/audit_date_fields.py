# Scans your diagnosis + instrument CSVs and recommends the right date columns.
from pathlib import Path
import sys, json, re
import pandas as pd

# Make project imports work when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_yaml, project_paths

EVENT_CANDIDATES = ["EVENT_ID", "EVENTID", "VISIT", "VISCOD", "VISIT_CODE", "EVENT_NAME"]
DIAG_DATE_CANDIDATES_DEFAULT = ["PDDXDT","DIAGNOSIS_DATE","DX_DATE","EVENT_DATE","EXAMDATE","DXDT"]
EXAM_DATE_CANDIDATES_DEFAULT = ["EXAMDATE","EXAM_DATE","VISIT_DATE","VISDT","SAMPLE_DATE","DRAW_DATE","COLLDATE","RUNDATE"]

def list_date_like_columns(df):
    cols = list(df.columns)
    return [c for c in cols if re.search(r"(date|dt)$", c, flags=re.I) or re.search(r"(date|dt)", c, flags=re.I)]

def summarize_dates(df, cols):
    out = []
    for c in cols:
        s = pd.to_datetime(df[c], errors="coerce")
        out.append({
            "column": c,
            "non_null": int(s.notna().sum()),
            "min": str(s.min()) if s.notna().any() else None,
            "max": str(s.max()) if s.notna().any() else None
        })
    out.sort(key=lambda x: (-x["non_null"], x["column"]))
    return out

def safe_read_csv(path):
    return pd.read_csv(path, low_memory=False, encoding_errors="ignore")

def main():
    paths = project_paths()
    cfg = load_yaml("configs/data.yaml")
    raw_dir = Path(paths["raw_dir"])
    files = cfg["files"]

    report = {"diagnosis": {}, "instruments": []}

    # ---- Diagnosis audit ----
    dx_path = raw_dir / files["diagnosis_events"]
    dx = safe_read_csv(dx_path)
    dx_date_cols = [c for c in DIAG_DATE_CANDIDATES_DEFAULT if c in dx.columns] or list_date_like_columns(dx)
    dx_sum = summarize_dates(dx, dx_date_cols)
    report["diagnosis"]["file"] = str(dx_path)
    report["diagnosis"]["columns_considered"] = dx_date_cols
    report["diagnosis"]["summary"] = dx_sum
    report["diagnosis"]["recommended_date_col"] = dx_sum[0]["column"] if dx_sum else None

    # ---- Instruments audit (for baseline) ----
    inst_list = files.get("instruments_optional", [])
    if isinstance(inst_list, str):
        inst_list = [inst_list]
    for rel in inst_list:
        p = raw_dir / rel
        if not p.exists():
            continue
        df = safe_read_csv(p)
        event_cols = [c for c in EVENT_CANDIDATES if c in df.columns]
        exam_cols = [c for c in EXAM_DATE_CANDIDATES_DEFAULT if c in df.columns] or list_date_like_columns(df)

        rec_event = event_cols[0] if event_cols else None
        exam_sum = summarize_dates(df, exam_cols)
        rec_exam = exam_sum[0]["column"] if exam_sum else None

        bl_count = None
        if rec_event:
            bl_count = int((df[rec_event] == cfg.get("baseline_event_code", "BL")).sum())

        report["instruments"].append({
            "file": str(p),
            "event_candidates_present": event_cols,
            "exam_candidates_present": exam_cols,
            "recommended_event_col": rec_event,
            "recommended_exam_date_col": rec_exam,
            "baseline_rows_found": bl_count
        })

    out_json = Path(paths["reports_dir"]) / "date_field_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Pretty console print
    print("\n=== Diagnosis file ===")
    print("File:", report["diagnosis"]["file"])
    print("Considered date columns:", report["diagnosis"]["columns_considered"])
    for row in report["diagnosis"]["summary"][:8]:
        print(f"- {row['column']}: non_null={row['non_null']}, min={row['min']}, max={row['max']}")
    print("Recommended diagnosis date column:", report["diagnosis"]["recommended_date_col"])

    print("\n=== Instrument files (baseline) ===")
    for inst in report["instruments"]:
        print("\nFile:", inst["file"])
        print("Event columns present:", inst["event_candidates_present"])
        print("Exam/date columns present:", inst["exam_candidates_present"][:6])
        print("Recommended event col:", inst["recommended_event_col"])
        print("Recommended exam date col:", inst["recommended_exam_date_col"])
        print("Rows with BL:", inst["baseline_rows_found"])

    print(f"\nSaved detailed JSON: {out_json}")

if __name__ == "__main__":
    main()
