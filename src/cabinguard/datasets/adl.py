"""ADL Dataset adapter — role TBD after inspection."""
from __future__ import annotations
import logging
import pandas as pd
from .base import CabinguardDatasetAdapter, DatasetStatus

logger = logging.getLogger(__name__)


class ADLAdapter(CabinguardDatasetAdapter):
    dataset_id = "adl"
    dataset_name = "ADL Dataset"
    cabinguard_role = "activity_posture_temporal_TBD"
    kaggle_dataset = "mocasbu/adl-dataset"

    def inspect(self) -> dict:
        report = {
            "dataset_id": self.dataset_id, "kaggle_dataset": self.kaggle_dataset,
            "cabinguard_role": "TO_BE_DETERMINED_AFTER_INSPECTION",
            "local_path": str(self.local_path), "status": DatasetStatus.NOT_DOWNLOADED,
            "file_count": "NOT_AVAILABLE", "size_mb": "NOT_AVAILABLE",
            "modalities": "NOT_AVAILABLE", "labels": "NOT_AVAILABLE", "class_counts": {},
            "notes": "Do not assume contents. Inspect file structure and labels first.",
        }
        if not self.local_path.exists():
            self._inspection_report = report
            return report
        files = [f for f in self.local_path.rglob("*") if f.is_file()]
        if not files:
            self._inspection_report = report
            return report
        report["file_count"] = len(files)
        report["size_mb"] = round(sum(f.stat().st_size for f in files) / 1e6, 2)
        exts = {}
        for f in files:
            exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
        report["file_extensions"] = exts
        report["directory_names"] = sorted(set(f.parent.name for f in files))[:30]
        for f in files:
            if f.suffix == ".csv":
                try:
                    df = pd.read_csv(f, nrows=3)
                    report["sample_csv_columns"] = list(df.columns)
                    break
                except Exception:
                    pass
        modalities = []
        if exts.get(".csv") or exts.get(".txt"): modalities.append("tabular")
        if exts.get(".mp4") or exts.get(".avi"): modalities.append("video")
        if exts.get(".mat") or exts.get(".npy"): modalities.append("signal")
        report["modalities"] = modalities or ["NOT_AVAILABLE"]
        report["status"] = DatasetStatus.AVAILABLE
        self._status = DatasetStatus.AVAILABLE
        self._inspection_report = report
        return report

    def build_manifest(self) -> pd.DataFrame:
        self.inspect()
        rows = []
        if self.local_path.exists():
            for ext in [".csv", ".mp4", ".avi", ".jpg", ".png", ".mat", ".npy"]:
                for f in self.local_path.rglob(f"*{ext}"):
                    rows.append({"sample_id": f"adl_{f.stem}", "subject_id": "NOT_AVAILABLE",
                                 "session_id": "NOT_AVAILABLE", "file_path": str(f),
                                 "label_raw": f.parent.name, "label_verified": False,
                                 "dataset_id": self.dataset_id, "modality": "NOT_AVAILABLE", "split": None})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["sample_id","subject_id","session_id","file_path","label_raw","label_verified","dataset_id","modality","split"])
        self._manifest = df
        return df

