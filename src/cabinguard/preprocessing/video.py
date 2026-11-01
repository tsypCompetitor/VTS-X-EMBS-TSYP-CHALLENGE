"""Video feature extraction — uses MediaPipe pose (frozen, cached)."""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def try_import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        logger.warning("MediaPipe not installed. pip install mediapipe")
        return None


class VideoFeatureExtractor:
    """
    Extract pose/visual features from video frames.
    Features are cached — raw video is NOT repeatedly loaded.
    Missing features are np.nan, NOT zero.
    Backbone is frozen by default.
    """

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe
        self._mp = try_import_mediapipe() if use_mediapipe else None
        self._pose = None
        if self._mp is not None:
            self._pose = self._mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.5)

    def extract_from_frame(self, frame_bgr: np.ndarray) -> dict:
        feat = {"head_pitch": np.nan, "head_roll": np.nan,
                "head_velocity": np.nan, "eye_closure": np.nan,
                "pose_confidence": np.nan, "extractor": "NOT_AVAILABLE"}
        if self._pose is None:
            return feat
        try:
            import cv2
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                nose, lear, rear = lm[0], lm[7], lm[8]
                feat["head_roll"] = float(np.degrees(np.arctan2(rear.y - lear.y, rear.x - lear.x)))
                feat["head_pitch"] = float((nose.y - (lear.y + rear.y) / 2) * 100)
                feat["pose_confidence"] = float(np.mean([nose.visibility, lear.visibility, rear.visibility]))
                feat["extractor"] = "mediapipe_pose"
        except Exception as e:
            logger.debug(f"Frame extraction failed: {e}")
        return feat

    def extract_from_video(self, video_path, max_frames=None, frame_step=1) -> list[dict]:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results = []
        idx = 0
        prev_pitch = None
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % frame_step == 0:
                feat = self.extract_from_frame(frame)
                feat["frame_idx"] = idx
                feat["timestamp_s"] = idx / fps
                if prev_pitch is not None and not np.isnan(feat.get("head_pitch", np.nan)):
                    feat["head_velocity"] = abs(feat["head_pitch"] - prev_pitch) * fps / frame_step
                if not np.isnan(feat.get("head_pitch", np.nan)):
                    prev_pitch = feat["head_pitch"]
                results.append(feat)
                if max_frames and len(results) >= max_frames: break
            idx += 1
        cap.release()
        return results

    def close(self):
        if self._pose: self._pose.close()

