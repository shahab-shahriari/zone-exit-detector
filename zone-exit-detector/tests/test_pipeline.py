"""End-to-end test on a synthetic video, using the model-free motion detector."""
import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from zone_exit.pipeline import EventSink, load_config, run

W, H, FPS = 320, 180, 25


def make_video(path, direction=+1):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(70):
        frame = np.zeros((H, W, 3), np.uint8)
        x = (-90 + 6 * i) if direction > 0 else (W + 10 - 6 * i)
        cv2.rectangle(frame, (x, 70), (x + 80, 120), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.zone = os.path.join(self.tmp, "zone.json")
        with open(self.zone, "w") as f:
            json.dump({"points": [[0.25, 0.2], [0.85, 0.2], [0.85, 0.9], [0.25, 0.9]]}, f)

    def config(self, direction):
        cfg = load_config(None)
        cfg["zone_file"] = self.zone
        cfg["detector"] = {"type": "motion", "motion": {"min_area": 1500}}
        cfg["rule"] = {"direction": direction, "min_distance_px": 40, "min_time_in_zone_s": 0.2}
        return cfg

    def test_vehicle_moving_in_allowed_direction_is_reported_once(self):
        video = os.path.join(self.tmp, "ltr.avi")
        make_video(video, +1)
        events = run(self.config("left_to_right"), video, sink=EventSink())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "zone_exit")
        self.assertGreater(events[0]["dx"], 40)

    def test_vehicle_moving_the_other_way_is_not_reported(self):
        video = os.path.join(self.tmp, "rtl.avi")
        make_video(video, -1)
        events = run(self.config("left_to_right"), video, sink=EventSink())
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
