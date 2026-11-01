"""
CABINGUARD-ADI Canonical Data Schema.
Every processed sample MUST carry complete provenance metadata.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

SCENARIO_CLASSES = {0: "NOMINAL", 1: "SYNCOPE", 2: "SEIZURE", 3: "COGNITIVE_AUTONOMIC", 4: "DISTRACTOR"}
SCENARIO_PHASES = ["NOMINAL", "PRODROMAL", "ACUTE"]

CANONICAL_FEATURES = [
    "head_pitch", "head_roll", "head_velocity", "wrist_velocity",
    "eye_closure", "perclos", "gaze_stability", "yawn_probability",
    "cop_y", "lumbar_contact", "quadrant_entropy",
    "heart_rate", "respiratory_rate", "ibi_variability", "bcg_quality",
    "steering_torque", "steering_torque_rate",
    "source_availability", "conflict_flag",
]
N_FEATURES = len(CANONICAL_FEATURES)


@dataclass
class SampleProvenance:
    """Mandatory provenance record for every processed sample."""
    source_dataset: str
    source_sample_id: str
    source_subject_id: Optional[str]
    source_session_id: Optional[str]
    is_synthetic: bool
    scenario_class: int
    scenario_phase: str
    injected_features: list
    transformation_parameters: dict
    split: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleProvenance":
        return cls(**d)

    def validate(self) -> None:
        assert self.scenario_class in SCENARIO_CLASSES, f"Invalid scenario_class {self.scenario_class}"
        assert self.scenario_phase in SCENARIO_PHASES, f"Invalid scenario_phase {self.scenario_phase}"

