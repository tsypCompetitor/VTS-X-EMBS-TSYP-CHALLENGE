"""
Radar Vital Signs Dataset adapter.
IMPORTANT: This is NOT a driver incapacitation dataset.
Use ONLY for cardiorespiratory signal processing experiments.
"""
from __future__ import annotations
import logging
import pandas as pd
from .base import CabinguardDatasetAdapter, DatasetStatus

logger = logging.getLogger(__name__)
_ROLE_WARNING = ("CRITICAL: cardiorespiratory signal processing ONLY. "
                 "Does NOT contain driver incapacitation events.")


class RadarVitalsAdapter(CabinguardDatasetAdapter):
    dataset_id = "radar_vitals"
    dataset_name = "Radar-Based Vital Sign Monitoring"
    cabinguard_role = "cardiorespiratory_signal_processing"
    kaggle_dataset = "cybersimar08/radar-based-vital-sign-monitoring"

    def inspect(self) -> dict:
        report = {
            "dataset_id": self.dataset_id, "kaggle_dataset": self.kaggle_dataset,
            "role_warning": _ROLE_WARNING, "local_path": str(self.local_path),
            "status": DatasetStatus.NOT_DOWNLOADED, "file_count": "NOT_AVAILABLE",
            "size_mb": "NOT_AVAILABLE", "signal_sampling_frequency": "NOT_AVAILABLE",
            "supported_use": ["heart rate extraction experiments", "respiratory rate experiments",
                              "BCG/SCG signal processing pipeline testing"],
            "unsupported_claims": ["Cannot validate driver incapacitation detection",
                                   "Cannot validate syncope/seizure scenarios"],
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
        for f in files:
            if f.suffix == ".csv":
                try:
                    df = pd.read_csv(f, nrows=3)
                    report["sample_columns"] = list(df.columns)
                    break
                except Exception:
                    pass
        report["status"] = DatasetStatus.AVAILABLE
        self._status = DatasetStatus.AVAILABLE
        self._inspection_report = report
        return report

    def build_manifest(self) -> pd.DataFrame:
        self.inspect()
        rows = []
        if self.local_path.exists():
            for ext in [".csv", ".mat", ".npy", ".npz"]:
                for f in self.local_path.rglob(f"*{ext}"):
                    rows.append({"sample_id": f"radar_{f.stem}", "subject_id": "NOT_AVAILABLE",
                                 "session_id": "NOT_AVAILABLE", "file_path": str(f),
                                 "label_raw": "vital_sign_signal", "dataset_id": self.dataset_id,
                                 "modality": "radar_signal", "split": None,
                                 "role_warning": "cardiorespiratory_only"})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["sample_id","subject_id","session_id","file_path","label_raw","dataset_id","modality","split","role_warning"])
        self._manifest = df
        return df

