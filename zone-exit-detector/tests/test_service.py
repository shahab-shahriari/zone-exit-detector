"""Service tests: auth, validation, session lifecycle, limits, timeouts."""
import json
import os
import tempfile
import time
import unittest

import cv2
import numpy as np

from zone_exit.pipeline import load_config
from zone_exit.service import SessionManager, create_app

W, H, FPS = 320, 180, 25
TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class CollectingSink:
    def __init__(self):
        self.events = []

    def emit(self, payload):
        self.events.append(payload)


class SlowDetector:
    """Never detects anything and takes a while per frame, so a session stays running."""

    def detect(self, frame):
        time.sleep(0.03)
        return []


def make_video(path):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(70):
        frame = np.zeros((H, W, 3), np.uint8)
        x = -90 + 6 * i
        cv2.rectangle(frame, (x, 70), (x + 80, 120), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        zone = os.path.join(tmp, "zone.json")
        with open(zone, "w") as f:
            json.dump({"points": [[0.25, 0.2], [0.85, 0.2], [0.85, 0.9], [0.25, 0.9]]}, f)
        video = os.path.join(tmp, "gate.avi")
        make_video(video)
        self.config = load_config(None)
        self.config["cameras"] = {"gate1": {"source": video, "zone_file": zone,
                                            "rule": {"direction": "left_to_right",
                                                     "min_distance_px": 40, "min_time_in_zone_s": 0.2}}}
        self.config["api"].update(max_sessions=1, default_timeout_s=30)
        self.sink = CollectingSink()

    def client(self, detector_factory=None):
        manager = SessionManager(self.config, self.sink, detector_factory)
        return create_app(self.config, manager, token=TOKEN).test_client()

    def wait_for(self, client, call_id, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            data = client.get(f"/sessions/{call_id}", headers=AUTH).get_json()
            if data["state"] != "running":
                return data
            time.sleep(0.05)
        self.fail("session did not finish in time")

    def test_health_needs_no_token_but_sessions_do(self):
        c = self.client()
        self.assertEqual(c.get("/health").status_code, 200)
        self.assertEqual(c.get("/sessions/x").status_code, 401)
        self.assertEqual(c.get("/sessions/x", headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_input_validation(self):
        c = self.client()
        post = lambda body: c.post("/sessions", json=body, headers=AUTH).status_code
        self.assertEqual(post({}), 400)
        self.assertEqual(post({"call_id": "bad id!", "camera_id": "gate1"}), 400)
        self.assertEqual(post({"call_id": "1", "camera_id": "gate1", "timeout_s": -5}), 400)
        self.assertEqual(post({"call_id": "1", "camera_id": "nope"}), 404)

    def test_detection_is_reported_with_call_id(self):
        c = self.client()
        r = c.post("/sessions", json={"call_id": "call-1", "camera_id": "gate1"}, headers=AUTH)
        self.assertEqual(r.status_code, 202)
        data = self.wait_for(c, "call-1")
        self.assertEqual(data["state"], "detected")
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(self.sink.events[0]["call_id"], "call-1")
        self.assertNotIn("source", json.dumps(data))  # stream addresses are never returned

    def test_duplicate_stop_and_capacity(self):
        c = self.client(lambda cfg: SlowDetector())
        self.assertEqual(c.post("/sessions", json={"call_id": "a", "camera_id": "gate1"}, headers=AUTH).status_code, 202)
        self.assertEqual(c.post("/sessions", json={"call_id": "a", "camera_id": "gate1"}, headers=AUTH).status_code, 409)
        self.assertEqual(c.post("/sessions", json={"call_id": "b", "camera_id": "gate1"}, headers=AUTH).status_code, 429)
        self.assertEqual(c.delete("/sessions/a", headers=AUTH).status_code, 202)
        self.assertEqual(self.wait_for(c, "a")["state"], "stopped")
        # the slot is free again
        self.assertEqual(c.post("/sessions", json={"call_id": "b", "camera_id": "gate1"}, headers=AUTH).status_code, 202)
        c.delete("/sessions/b", headers=AUTH)
        self.wait_for(c, "b")

    def test_timeout_emits_event(self):
        c = self.client(lambda cfg: SlowDetector())
        c.post("/sessions", json={"call_id": "t", "camera_id": "gate1", "timeout_s": 0.3}, headers=AUTH)
        self.assertEqual(self.wait_for(c, "t")["state"], "timeout")
        self.assertEqual([e["event"] for e in self.sink.events], ["timeout"])

    def test_unknown_session_is_404(self):
        self.assertEqual(self.client().get("/sessions/nothing", headers=AUTH).status_code, 404)


if __name__ == "__main__":
    unittest.main()
