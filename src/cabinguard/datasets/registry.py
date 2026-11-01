"""Dataset registry for CABINGUARD-ADI."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict
from .base import CabinguardDatasetAdapter
from .huggingface_driver import HuggingFaceDriverAdapter
from .ala_mahlak import AlaMahlakAdapter
from .radar_vitals import RadarVitalsAdapter
from .adl import ADLAdapter
from .adms import ADMSAdapter

logger = logging.getLogger(__name__)

ADAPTER_CLASSES = {
    "hf_driver": HuggingFaceDriverAdapter,
    "ala_mahlak": AlaMahlakAdapter,
    "radar_vitals": RadarVitalsAdapter,
    "adl": ADLAdapter,
    "adms": ADMSAdapter,
}


def build_registry(datasets_config: dict, data_root: str | Path) -> Dict[str, CabinguardDatasetAdapter]:
    data_root = Path(data_root)
    registry = {}
    for dataset_id, cls in ADAPTER_CLASSES.items():
        ds_cfg = datasets_config.get(dataset_id, {})
        if not ds_cfg.get("enabled", True):
            continue
        local_path = ds_cfg.get("local_path", data_root / "raw" / dataset_id)
        adapter = cls(local_path=local_path, config=ds_cfg)
        registry[dataset_id] = adapter
        logger.debug(f"Registered: {dataset_id}")
    return registry

