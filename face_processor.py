"""
Face Analysis Processor
=======================
Core processing module for a Qoves-style facial analysis app.

Uses the modern MediaPipe Tasks Face Landmarker.
Calculates facial thirds, symmetry, eye spacing, nose width, lip ratio, jaw angles, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARKS = {
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
    "forehead_top": 10,
    "left_temple": 54,
    "right_temple": 284,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "nose_tip": 1,
    "nose_bridge": 6,
    "nose_left": 98,
    "nose_right": 327,
    "nose_bottom": 2,
    "upper_lip_top": 13,
    "lower_lip_bottom": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip_center": 0,
    "lower_lip_center": 17,
    "left_brow_inner": 107,
    "left_brow_outer": 70,
    "right_brow_inner": 336,
    "right_brow_outer": 300,
    "left_jaw": 172,
    "right_jaw": 397,
}


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0


@dataclass
class MeasurementResult:
    landmarks: Dict[str, Point]
    facial_thirds: Dict[str, float]
    horizontal_fifths: Dict[str, float]
    symmetry_score: float
    eye_spacing_ratio: float
    interocular_distance: float
    nose_width_ratio: float
    lip_ratio: float
    jaw_angle_left: Optional[float]
    jaw_angle_right: Optional[float]
    face_width_to_height: float
    midface_ratio: float
    scores: Dict[str, float]
    image_width: int
    image_height: int
    success: bool
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["landmarks"] = {k: {"x": v.x, "y": v.y, "z": v.z} for k, v in self.landmarks.items()}
        return d


class FaceProcessor:
    def __init__(self, model_path: str | Path | None = None, num_faces: int = 1):
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "face_landmarker.task"
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Download from:\n"
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _load_image(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        return img_bgr, mp_image

    def _get_landmarks(self, mp_image: mp.Image) -> Optional[List[Point]]:
        detection = self.landmarker.detect(mp_image)
        if not detection.face_landmarks:
            return None
        face = detection.face_landmarks[0]
        return [Point(x=lm.x, y=lm.y, z=lm.z) for lm in face]

    def _dist(self, p1: Point, p2: Point) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _angle(self, a: Point, b: Point, c: Point) -> float:
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        cosine = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba) * math.hypot(*bc) + 1e-8)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    def analyze(self, front_path: str | Path, profile_left=None, profile_right=None) -> MeasurementResult:
        img_bgr, mp_image = self._load_image(front_path)
        h, w = img_bgr.shape[:2]

        all_points = self._get_landmarks(mp_image)
        if all_points is None:
            return MeasurementResult(
                landmarks={}, facial_thirds={}, horizontal_fifths={},
                symmetry_score=0.0, eye_spacing_ratio=0.0, interocular_distance=0.0,
                nose_width_ratio=0.0, lip_ratio=0.0, jaw_angle_left=None, jaw_angle_right=None,
                face_width_to_height=0.0, midface_ratio=0.0, scores={},
                image_width=w, image_height=h, success=False,
                message="No face detected. Use a clear front-facing photo with good lighting."
            )

        lm: Dict[str, Point] = {}
        for name, idx in LANDMARKS.items():
            if idx < len(all_points):
                lm[name] = all_points[idx]

        # Facial thirds
        forehead = lm.get("forehead_top")
        left_brow = lm.get("left_brow_inner") or lm.get("left_brow_outer")
        right_brow = lm.get("right_brow_inner") or lm.get("right_brow_outer")
        nose_bottom = lm.get("nose_bottom") or lm.get("nose_tip")
        chin = lm.get("chin")

        facial_thirds = {}
        if forehead and left_brow and nose_bottom and chin:
            brow_y = (left_brow.y + right_brow.y)/2 if right_brow else left_brow.y
            upper = brow_y - forehead.y
            middle = nose_bottom.y - brow_y
            lower = chin.y - nose_bottom.y
            total = upper + middle + lower + 1e-8
            facial_thirds = {
                "upper_third": round(upper/total, 4),
                "middle_third": round(middle/total, 4),
                "lower_third": round(lower/total, 4),
                "ideal_upper": 0.3333, "ideal_middle": 0.3333, "ideal_lower": 0.3333,
            }

        left_cheek = lm.get("left_cheek")
        right_cheek = lm.get("right_cheek")
        horizontal_fifths = {}
        if left_cheek and right_cheek:
            horizontal_fifths = {"face_width": round(self._dist(left_cheek, right_cheek), 4)}

        # Symmetry
        symmetry_pairs = [
            ("left_eye_outer", "right_eye_outer"),
            ("left_eye_inner", "right_eye_inner"),
            ("left_brow_outer", "right_brow_outer"),
            ("mouth_left", "mouth_right"),
            ("left_cheek", "right_cheek"),
        ]
        diffs = []
        midline_x = lm["nose_tip"].x if "nose_tip" in lm else 0.5
        for ln, rn in symmetry_pairs:
            if ln in lm and rn in lm:
                diffs.append(abs(abs(lm[ln].x - midline_x) - abs(lm[rn].x - midline_x)))
        symmetry_score = 100.0
        if diffs:
            symmetry_score = max(0.0, min(100.0, 100.0 - (sum(diffs)/len(diffs)*500)))

        # Eyes
        interocular = 0.0
        eye_spacing_ratio = 0.0
        lei, rei = lm.get("left_eye_inner"), lm.get("right_eye_inner")
        leo, reo = lm.get("left_eye_outer"), lm.get("right_eye_outer")
        if lei and rei and leo and reo:
            interocular = self._dist(lei, rei)
            avg_eye = (self._dist(leo, lei) + self._dist(reo, rei)) / 2
            eye_spacing_ratio = interocular / (avg_eye + 1e-8)

        # Nose
        nose_width_ratio = 0.0
        if "nose_left" in lm and "nose_right" in lm and left_cheek and right_cheek:
            nose_width_ratio = self._dist(lm["nose_left"], lm["nose_right"]) / (self._dist(left_cheek, right_cheek) + 1e-8)

        # Lips
        lip_ratio = 0.0
        if "upper_lip_top" in lm and "lower_lip_bottom" in lm and "upper_lip_center" in lm:
            upper_h = abs(lm["upper_lip_top"].y - lm["upper_lip_center"].y)
            lower_h = abs(lm["lower_lip_bottom"].y - lm["upper_lip_center"].y)
            if upper_h > 1e-6:
                lip_ratio = lower_h / upper_h

        # Jaw angles
        jaw_angle_left = jaw_angle_right = None
        if "left_jaw" in lm and "chin" in lm and "left_cheek" in lm:
            jaw_angle_left = self._angle(lm["left_cheek"], lm["left_jaw"], lm["chin"])
        if "right_jaw" in lm and "chin" in lm and "right_cheek" in lm:
            jaw_angle_right = self._angle(lm["right_cheek"], lm["right_jaw"], lm["chin"])
