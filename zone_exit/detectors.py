"""Detectors return a list of (x1, y1, x2, y2, confidence) tuples per frame."""
import threading
from typing import List, Sequence, Tuple

import cv2

Detection = Tuple[float, float, float, float, float]


class MotionDetector:
    """Background subtraction. Needs no model, so it is handy for a first test.
    It finds anything that moves, not only vehicles."""

    def __init__(self, min_area: int = 1500, history: int = 200, var_threshold: float = 32.0):
        self.min_area = min_area
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame) -> List[Detection]:
        mask = self._bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= self.min_area:
                out.append((float(x), float(y), float(x + w), float(y + h), 1.0))
        return out


class YoloDetector:
    """Ultralytics YOLO. Default classes are the COCO ids for car, bus and truck."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.3, iou: float = 0.5,
                 classes: Sequence[int] = (2, 5, 7), device=None):
        from ultralytics import YOLO  # imported here so the motion detector works without it

        self._model = YOLO(weights)
        self._kwargs = dict(conf=conf, iou=iou, classes=list(classes), verbose=False)
        if device not in (None, "auto"):
            self._kwargs["device"] = device

    def detect(self, frame) -> List[Detection]:
        result = self._model.predict(frame, **self._kwargs)[0]
        out = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append((x1, y1, x2, y2, float(box.conf[0])))
        return out


def make_detector(cfg: dict):
    kind = cfg.get("type", "motion")
    if kind == "yolo":
        return YoloDetector(**cfg.get("yolo", {}))
    if kind == "motion":
        return MotionDetector(**cfg.get("motion", {}))
    raise ValueError(f"unknown detector type: {kind!r}")


class LockedDetector:
    """Lets several threads share one detector (for example one YOLO model in memory)."""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def detect(self, frame):
        with self._lock:
            return self._inner.detect(frame)
