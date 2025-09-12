from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_paths() -> Dict[str, str]:
    """Load configs/paths.yaml and return resolved paths as strings."""
    cfg = load_yaml("configs/paths.yaml")
    # normalize
    for k in list(cfg.keys()):
        if k.endswith("_dir") or k.endswith("_root"):
            cfg[k] = str(Path(cfg[k]).resolve())
    return cfg
