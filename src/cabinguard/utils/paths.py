"""Path resolution helpers."""
import os
from pathlib import Path


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def get_data_root() -> Path:
    root = os.environ.get("CABINGUARD_DATA_ROOT")
    return Path(root) if root else get_project_root() / "data"


def get_output_root() -> Path:
    root = os.environ.get("CABINGUARD_OUTPUT_ROOT")
    return Path(root) if root else get_project_root() / "outputs"

