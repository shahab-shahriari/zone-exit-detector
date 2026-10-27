"""Small geometry helpers (no external dependencies)."""
from typing import Sequence, Tuple

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray-casting test. Works for concave polygons."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


def denormalize(points: Sequence[Point], width: int, height: int):
    """Convert 0..1 coordinates to pixel coordinates for a given frame size."""
    return [(px * width, py * height) for px, py in points]


def box_center(box: Box) -> Point:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def iou(a: Box, b: Box) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
