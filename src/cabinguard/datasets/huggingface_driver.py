"""HuggingFace In-Cabin Driver Dataset adapter."""
from __future__ import annotations
import os, logging
import pandas as pd
from pathlib import Path
from .base import CabinguardDatasetAdapter, DatasetStatus

logger = logging.getLogger(__name__)
HF_REPO = "bigamma/In_Cabin_Behavior_Driver_safe_drive"


class HuggingFaceDriverAdapter(CabinguardDatasetAdapter):
    dataset_id = "hf_driver"
    dataset_name = "In-Cabin Behavior Driver Safe Drive (HuggingFace)"
    cabinguard_role = "visual_baseline"

    def __init__(self, local_path, config=None):
        super().__init__(local_path, config)
        self.hf_token = os.environ.get("HF_TOKEN")

    def download(self) -> bool:
        if not self.hf_token:
            logger.warning("HF_TOKEN not set. Dataset may be gated.")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                              local_dir=str(self.local_path), token=self.hf_token)
            return True
        except Exception as e:
            logger.warning(f"HF dataset unavailable: {e}")
            self._status = DatasetStatus.UNAVAILABLE
            return False

    def inspect(self) -> dict:
        report = {
            "dataset_id": self.dataset_id, "hf_repo": HF_REPO,
            "cabinguard_role": self.cabinguard_role,
            "status": DatasetStatus.NOT_DOWNLOADED,
            "local_path": str(self.local_path),
            "file_count": "NOT_AVAILABLE", "size_mb": "NOT_AVAILABLE",
            "modalities": ["video (expected)"], "subjects": "NOT_AVAILABLE",
            "labels": "NOT_AVAILABLE", "class_counts": {},
            "notes": "Access may be gated. Set HF_TOKEN. Inspect labels before assigning meaning.",
            "unsupported_claims": ["Do NOT assume medical labels without verification."],
            "possible_cabinguard_use": ["seated in-cabin geometry", "visual driver behavior baseline"],
        }
        if not self.local_path.exists():
            self._inspection_report = report
            return report
        files = [f for f in self.local_path.rglob("*") if f.is_file()]
        report["file_count"] = len(files)
        report["size_mb"] = round(sum(f.stat().st_size for f in files) / 1e6, 2)
        exts = {}
        for f in files:
            exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
        report["file_extensions"] = exts
        report["status"] = DatasetStatus.AVAILABLE if files else DatasetStatus.NOT_DOWNLOADED
        self._status = report["status"]
        self._inspection_report = report
        return report

    def build_manifest(self) -> pd.DataFrame:
        self.inspect()
        rows = []
        if self.local_path.exists():
            for ext in [".mp4", ".avi", ".jpg", ".jpeg", ".png"]:
                for f in self.local_path.rglob(f"*{ext}"):
                    rows.append({"sample_id": f"hf_{f.stem}", "subject_id": "NOT_AVAILABLE",
                                 "session_id": "NOT_AVAILABLE", "file_path": str(f),
                                 "label_raw": "NOT_AVAILABLE", "dataset_id": self.dataset_id,
                                 "modality": "video" if ext in {".mp4",".avi"} else "image", "split": None})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["sample_id","subject_id","session_id","file_path","label_raw","dataset_id","modality","split"])
        self._manifest = df
        return df

