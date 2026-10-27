"""A minimal IoU tracker: greedy matching of boxes between frames."""
from typing import Dict, List

from .geometry import Box, iou


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # frames a track may go unmatched before removal
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}

    def update(self, boxes: List[Box]) -> Dict[int, Box]:
        """Match new boxes to existing tracks; return {track_id: box} for this frame."""
        pairs = []
        for tid, track in self._tracks.items():
            for j, box in enumerate(boxes):
                score = iou(track["box"], box)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, j))
        pairs.sort(reverse=True)  # best matches first, so results do not depend on dict order

        used_tracks, used_boxes, current = set(), set(), {}
        for _, tid, j in pairs:
            if tid in used_tracks or j in used_boxes:
                continue
            used_tracks.add(tid)
            used_boxes.add(j)
            self._tracks[tid].update(box=boxes[j], age=0)
            current[tid] = boxes[j]

        for tid in list(self._tracks):
            if tid not in used_tracks:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        for j, box in enumerate(boxes):
            if j not in used_boxes:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"box": box, "age": 0}
                current[tid] = box
        return current
