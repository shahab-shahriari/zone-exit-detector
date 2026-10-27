"""Read frames, detect, track, apply the zone rule, emit events."""
import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np
import requests
import yaml

from .detectors import make_detector
from .exit_logic import ExitRule, ZoneExitMonitor
from .geometry import denormalize
from .tracker import IoUTracker

DEFAULTS = {
    "detector": {"type": "motion"},
    "zone_file": "zone.json",
    "rule": {"direction": "any", "min_distance_px": 20, "min_time_in_zone_s": 0.5},
    "tracker": {"iou_threshold": 0.3, "max_age": 15},
    "frame_stride": 1,
    "events": {"jsonl": None, "webhook_url": None, "webhook_timeout_s": 3.0},
    "cameras": {},
    "api": {"max_sessions": 4, "default_timeout_s": 120, "retention_s": 600, "token_env": "ZONE_EXIT_TOKEN"},
}


def merge_config(base: dict, extra: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str]) -> dict:
    if not path:
        return copy.deepcopy(DEFAULTS)
    with open(path, "r", encoding="utf-8") as f:
        return merge_config(DEFAULTS, yaml.safe_load(f))


def load_zone(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [tuple(p) for p in json.load(f)["points"]]


def camera_config(config: dict, camera_id: str):
    """Return (config for one camera, expanded source). Camera settings override the globals.
    The source may reference environment variables (e.g. ${GATE1_URL}) so that stream
    addresses and passwords stay out of files."""
    cam = config["cameras"][camera_id]
    cfg = {k: v for k, v in config.items() if k != "cameras"}
    overrides = {k: v for k, v in cam.items() if k in ("zone_file", "rule", "tracker", "frame_stride")}
    cfg = merge_config(cfg, overrides)
    source = os.path.expandvars(str(cam["source"]))
    if "$" in source:
        raise RuntimeError(f"camera {camera_id!r}: an environment variable in 'source' is not set")
    return cfg, source


class EventSink:
    """Prints events, optionally appends them to a JSON-lines file and posts them to a webhook.
    The webhook call runs in a thread, so a slow server never blocks frame processing."""

    def __init__(self, jsonl_path=None, webhook_url=None, timeout=3.0):
        self.jsonl_path = jsonl_path
        self.webhook_url = webhook_url
        self.timeout = timeout
        self._file_lock = threading.Lock()

    def emit(self, payload: dict) -> None:
        print(json.dumps(payload))
        if self.jsonl_path:
            with self._file_lock, open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        if self.webhook_url:
            threading.Thread(target=self._post, args=(payload,), daemon=True).start()

    def _post(self, payload: dict) -> None:
        try:
            requests.post(self.webhook_url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            print(f"warning: webhook failed: {exc}")


def sink_from_config(config: dict) -> EventSink:
    ev = config["events"]
    return EventSink(ev.get("jsonl"), ev.get("webhook_url"), float(ev.get("webhook_timeout_s", 3.0)))


def is_live(source) -> bool:
    return isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))


def _reopen(source, stop_event, tries: int = 5, delay: float = 2.0):
    for attempt in range(1, tries + 1):
        print(f"warning: stream lost, reconnecting ({attempt}/{tries})")
        if stop_event is not None:
            if stop_event.wait(delay):
                return None
        else:
            time.sleep(delay)
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def _draw(frame, polygon_px, tracks, triggered):
    pts = np.array(polygon_px, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    for tid, (x1, y1, x2, y2) in tracks.items():
        color = (0, 0, 255) if tid in triggered else (0, 200, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"id {tid}", (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


@dataclass
class RunResult:
    events: List[dict]
    reason: str  # eof | stream_lost | stopped | deadline | first_event | closed | max_frames


def process(config: dict, source, *, show: bool = False, save_video: Optional[str] = None,
            max_frames: Optional[int] = None, sink=None, detector=None,
            stop_event: Optional[threading.Event] = None, max_seconds: Optional[float] = None,
            stop_after_first: bool = False, extra: Optional[dict] = None) -> RunResult:
    """Process one video source until it ends, is stopped, times out or (optionally) fires once."""
    detector = detector or make_detector(config["detector"])
    zone_norm = load_zone(config["zone_file"])
    rule = ExitRule(**config["rule"])
    tracker = IoUTracker(**config["tracker"])
    stride = max(1, int(config.get("frame_stride", 1)))
    sink = sink or sink_from_config(config)

    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("cannot open the video source")
    live = is_live(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    deadline = time.monotonic() + max_seconds if max_seconds else None

    monitor, writer, polygon_px = None, None, None
    events: List[dict] = []
    triggered = set()
    frame_idx = 0
    reason = "stream_lost" if live else "eof"
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                reason = "stopped"
                break
            if deadline is not None and time.monotonic() >= deadline:
                reason = "deadline"
                break

            ok, frame = cap.read()
            if not ok:
                if live:
                    cap.release()
                    cap = _reopen(source, stop_event)
                    if cap is not None:
                        continue
                    if stop_event is not None and stop_event.is_set():
                        reason = "stopped"
                break
            frame_idx += 1
            if frame_idx % stride:
                continue

            h, w = frame.shape[:2]
            if monitor is None:
                polygon_px = denormalize(zone_norm, w, h)
                monitor = ZoneExitMonitor(polygon_px, rule)

            now = time.monotonic() if live else frame_idx / fps
            boxes = [d[:4] for d in detector.detect(frame)]
            tracks = tracker.update(boxes)

            for ev in monitor.update(tracks, now):
                triggered.add(ev.track_id)
                payload = {
                    "event": "zone_exit",
                    "track_id": ev.track_id,
                    "frame": frame_idx,
                    "dx": round(ev.dx, 1),
                    "dy": round(ev.dy, 1),
                    "seconds_in_zone": round(ev.seconds_in_zone, 2),
                    "time": datetime.now(timezone.utc).isoformat(),
                }
                if not live:
                    payload["video_time_s"] = round(ev.timestamp, 2)
                payload.update(extra or {})
                events.append(payload)
                sink.emit(payload)

            if show or save_video:
                view = _draw(frame.copy(), polygon_px, tracks, triggered)
                if save_video and writer is None:
                    writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / stride, (w, h))
                if writer is not None:
                    writer.write(view)
                if show:
                    cv2.imshow("zone-exit", view)
                    if cv2.waitKey(1) & 0xFF == 27:  # Esc
                        reason = "closed"
                        break
            if stop_after_first and events:
                reason = "first_event"
                break
            if max_frames and frame_idx >= max_frames:
                reason = "max_frames"
                break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return RunResult(events, reason)


def run(config: dict, source, **kwargs) -> List[dict]:
    """Convenience wrapper that returns only the events."""
    return process(config, source, **kwargs).events
