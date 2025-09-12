# Avoid __future__ import ordering issues and make 'src' importable.

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.load_ppmi import ensure_unique_columns, drop_duplicate_id_columns
from src.utils.config import load_yaml, project_paths
from src.data.merge_ppmi import assemble_master

def main():
    paths = project_paths()
    cfg = load_yaml("configs/data.yaml")

    out_parquet = Path(paths["processed_dir"]) / "master_baseline.parquet"
    out_csv = Path(paths["processed_dir"]) / "master_baseline_preview.csv"

    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    master = assemble_master(cfg, paths)
    master = drop_duplicate_id_columns(master, cfg["id_column"])
    master = ensure_unique_columns(master)
    master = master.loc[:, ~master.columns.duplicated()]
    # Save artifacts
    master.to_parquet(out_parquet, index=False)
    master.head(200).to_csv(out_csv, index=False)

    print(f"✅ Built dataset: {out_parquet}")
    print(f"👀 Preview (first 200 rows): {out_csv}")
    print(f"Rows: {len(master):,}  |  Columns: {len(master.columns):,}")
    print("\nColumns:", ", ".join(map(str, master.columns[:30])), "...")

if __name__ == "__main__":
    main()
