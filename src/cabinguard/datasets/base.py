"""Base dataset adapter interface for CABINGUARD-ADI."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import logging, json, hashlib
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetStatus:
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_DOWNLOADED = "NOT_DOWNLOADED"


class CabinguardDatasetAdapter(ABC):
    dataset_id: str = ""
    dataset_name: str = ""
    cabinguard_role: str = ""

    def __init__(self, local_path: str | Path, config: dict | None = None):
        self.local_path = Path(local_path)
        self.config = config or {}
        self._status: str = DatasetStatus.NOT_DOWNLOADED
        self._manifest: pd.DataFrame | None = None
        self._inspection_report: dict | None = None

    @property
    def status(self) -> str:
        return self._status

    @abstractmethod
    def inspect(self) -> dict: ...

    @abstractmethod
    def build_manifest(self) -> pd.DataFrame: ...

    def get_manifest(self) -> pd.DataFrame:
        if self._manifest is None:
            raise RuntimeError(f"Call build_manifest() first for {self.dataset_id}")
        return self._manifest

    def manifest_hash(self) -> str:
        if self._manifest is None:
            return "NOT_AVAILABLE"
        content = self._manifest.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    def save_manifest(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.dataset_id}_manifest.parquet"
        self._manifest.to_parquet(path, index=False)
        logger.info(f"Manifest saved: {path} ({len(self._manifest)} rows)")
        return path

    def save_inspection_report(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.dataset_id}_inspection.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._inspection_report, f, indent=2, default=str)
        return path

