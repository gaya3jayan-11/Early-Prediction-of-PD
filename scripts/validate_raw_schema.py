# NOTE: this version supports list-valued entries (e.g., plasma_biomarkers: [..])
# and avoids __future__ import ordering issues.

from pathlib import Path
import sys, json
import pandas as pd

# Make 'src' importable when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_yaml, project_paths
from src.utils.data_checks import (
    read_table,
    describe_basic,
    check_required_columns,
    check_id_uniqueness,
)

def _iter_files(files_cfg):
    """
    Yield tuples of (logical_table_name, display_name, relative_path_string).
    Handles string or list entries under each key.
    """
    for table_name, value in files_cfg.items():
        if isinstance(value, list):
            for i, v in enumerate(value):
                yield f"{table_name}[{i}]", table_name, v
        elif isinstance(value, str):
            yield table_name, table_name, value
        else:
            # skip unknown types
            continue

def main() -> int:
    paths = project_paths()
    cfg = load_yaml("configs/data.yaml")

    raw_dir = Path(paths["raw_dir"])
    out_md = Path(paths["reports_dir"]) / "schema_report.md"
    out_json = Path(paths["reports_dir"]) / "schema_report.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    files_cfg = cfg.get("files", {})
    exp_cols = cfg.get("expected_columns", {})
    id_raw = cfg.get("join_key_raw", "PATNO")

    report = {"raw_dir": str(raw_dir), "tables": []}
    md_lines = [
        f"# Schema Report\n",
        f"- raw_dir: `{raw_dir}`\n",
        f"- id_raw: `{id_raw}`\n",
    ]

    for logical_name, base_name, relpath in _iter_files(files_cfg):
        item = {"table": logical_name, "base_group": base_name, "file": relpath, "exists": False}
        path = raw_dir / relpath if relpath and relpath != "<PUT_FILENAME_HERE>" else None

        if path and path.exists():
            item["exists"] = True
            try:
                df = read_table(path, nrows=10000)   # sample or full if small
                item["basic"] = describe_basic(df, logical_name)
                required = exp_cols.get(base_name, [])
                present, missing = check_required_columns(df, required)
                item["required_present"] = present
                item["required_missing"] = missing
                item["id_check"] = check_id_uniqueness(df, id_raw)
            except Exception as e:
                item["error"] = f"{type(e).__name__}: {e}"
        else:
            item["note"] = "Filename not set or file not found."

        report["tables"].append(item)

        # Append to MD
        md_lines.append(f"## {logical_name}")
        md_lines.append(f"- group: `{base_name}`")
        md_lines.append(f"- file: `{relpath}`")
        md_lines.append(f"- exists: `{item['exists']}`")
        if "error" in item:
            md_lines.append(f"- error: `{item['error']}`")
        if "basic" in item:
            shape = item["basic"]["shape"]
            md_lines.append(f"- shape (sample/full): `{shape}`")
            if item.get("required_missing"):
                md_lines.append(f"- missing required columns: `{item['required_missing']}`")
            idc = item.get("id_check", {})
            md_lines.append(
                f"- id `{id_raw}` present: `{idc.get('exists')}`, unique: `{idc.get('unique')}`, dupes: `{idc.get('dupe_count')}`"
            )
        md_lines.append("")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
