"""
ADMS Dataset adapter.
Verified behavioral classes: eye_closure, yawning, seatbelt, smoking, mobile_phone_use.
NEVER convert to medical diagnoses.
"""
from __future__ import annotations
import logging
import pandas as pd
from .base import CabinguardDatasetAdapter, DatasetStatus

logger = logging.getLogger(__name__)
VERIFIED_CLASSES = {"eye_closure", "yawning", "seatbelt", "smoking", "mobile_phone_use"}


class ADMSAdapter(CabinguardDatasetAdapter):
    dataset_id = "adms"
    dataset_name = "Advance Driver Monitoring System Dataset"
    cabinguard_role = "visual_distractor_confounders"
    kaggle_dataset = "smartcity12/adms-dataset"

    def inspect(self) -> dict:
        report = {
            "dataset_id": self.dataset_id, "kaggle_dataset": self.kaggle_dataset,
            "cabinguard_role": self.cabinguard_role, "local_path": str(self.local_path),
            "status": DatasetStatus.NOT_DOWNLOADED,
            "expected_classes": list(VERIFIED_CLASSES), "verified_classes": [],
            "file_count": "NOT_AVAILABLE", "size_mb": "NOT_AVAILABLE",
            "notes": "Use verified classes only. Never convert to medical diagnoses.",
            "unsupported_claims": [
                "Eye closure != loss of consciousness",
                "Yawning != seizure prodrome",
                "Labels cannot be directly interpreted as medical diagnoses"
            ],
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
        cls_file = self.local_path / "classes.txt"
        if cls_file.exists():
            classes = cls_file.read_text().strip().split("\n")
            report["yolo_classes"] = classes
            report["verified_classes"] = [c for c in classes if c.lower().replace(" ","_") in VERIFIED_CLASSES]
        modalities = []
        if exts.get(".jpg") or exts.get(".png"): modalities.append("image")
        if exts.get(".mp4") or exts.get(".avi"): modalities.append("video")
        if any(f.suffix == ".txt" for f in files): modalities.append("yolo_annotations")
        report["modalities"] = modalities or ["NOT_AVAILABLE"]
        report["status"] = DatasetStatus.AVAILABLE
        self._status = DatasetStatus.AVAILABLE
        self._inspection_report = report
        return report

    def build_manifest(self) -> pd.DataFrame:
        self.inspect()
        rows = []
        if self.local_path.exists():
            for ext in [".jpg", ".jpeg", ".png", ".mp4", ".avi"]:
                for f in self.local_path.rglob(f"*{ext}"):
                    ann = f.with_suffix(".txt")
                    rows.append({"sample_id": f"adms_{f.stem}", "subject_id": "NOT_AVAILABLE",
                                 "session_id": "NOT_AVAILABLE", "file_path": str(f),
                                 "annotation_path": str(ann) if ann.exists() else None,
                                 "label_raw": f.parent.name, "label_verified": False,
                                 "dataset_id": self.dataset_id,
                                 "modality": "video" if ext in {".mp4",".avi"} else "image", "split": None})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["sample_id","subject_id","session_id","file_path","annotation_path",
                     "label_raw","label_verified","dataset_id","modality","split"])
        self._manifest = df
        return df

