"""Decide when a tracked vehicle has started leaving a zone.

A vehicle triggers an event once, when it has spent at least `min_time_in_zone_s`
inside the zone and its centre has moved `min_distance_px` from where it first
appeared in the zone, in the allowed direction.

The monitor takes timestamps as arguments (no hidden clock), so it can be tested
with plain numbers.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .geometry import Box, Point, box_center, point_in_polygon

DIRECTIONS = ("any", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")


@dataclass(frozen=True)
class ExitRule:
    direction: str = "any"
    min_distance_px: float = 20.0
    min_time_in_zone_s: float = 0.5
    stale_after_s: float = 30.0

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")


@dataclass
class ExitEvent:
    track_id: int
    timestamp: float
    dx: float
    dy: float
    seconds_in_zone: float


class _TrackState:
    __slots__ = ("first_pos", "first_time", "triggered", "last_seen")

    def __init__(self, now: float):
        self.first_pos: Optional[Point] = None
        self.first_time: float = now
        self.triggered = False
        self.last_seen = now


def direction_ok(dx: float, dy: float, rule: ExitRule) -> bool:
    d, m = rule.direction, rule.min_distance_px
    if d == "left_to_right":
        return dx >= m
    if d == "right_to_left":
        return dx <= -m
    if d == "top_to_bottom":
        return dy >= m
    if d == "bottom_to_top":
        return dy <= -m
    return abs(dx) >= m or abs(dy) >= m


class ZoneExitMonitor:
    def __init__(self, polygon_px: Sequence[Point], rule: ExitRule):
        if len(polygon_px) < 3:
            raise ValueError("a zone needs at least 3 points")
        self.polygon = list(polygon_px)
        self.rule = rule
        self._states: Dict[int, _TrackState] = {}

    def update(self, tracks: Dict[int, Box], now: float) -> List[ExitEvent]:
        events: List[ExitEvent] = []
        for tid, box in tracks.items():
            cx, cy = box_center(box)
            state = self._states.setdefault(tid, _TrackState(now))
            state.last_seen = now

            if not point_in_polygon(cx, cy, self.polygon):
                state.first_pos = None  # re-entering the zone starts a fresh comparison
                continue

            if state.first_pos is None:
                state.first_pos = (cx, cy)
                state.first_time = now
                continue
            if state.triggered:
                continue

            dx, dy = cx - state.first_pos[0], cy - state.first_pos[1]
            seconds = now - state.first_time
            if seconds >= self.rule.min_time_in_zone_s and direction_ok(dx, dy, self.rule):
                state.triggered = True
                events.append(ExitEvent(tid, now, dx, dy, seconds))

        self._drop_stale(now)
        return events

    def _drop_stale(self, now: float) -> None:
        stale = [t for t, s in self._states.items() if now - s.last_seen > self.rule.stale_after_s]
        for t in stale:
            del self._states[t]
