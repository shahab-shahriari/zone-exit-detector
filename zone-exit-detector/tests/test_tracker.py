import unittest

from zone_exit.tracker import IoUTracker


class TrackerTests(unittest.TestCase):
    def test_id_is_kept_while_box_moves_slowly(self):
        t = IoUTracker(iou_threshold=0.3)
        first = t.update([(0, 0, 50, 50)])
        (tid,) = first
        for step in range(1, 6):
            current = t.update([(step * 5, 0, 50 + step * 5, 50)])
            self.assertEqual(list(current), [tid])

    def test_far_box_gets_new_id(self):
        t = IoUTracker()
        a = t.update([(0, 0, 50, 50)])
        b = t.update([(0, 0, 50, 50), (300, 300, 350, 350)])
        self.assertEqual(len(b), 2)
        self.assertIn(list(a)[0], b)

    def test_track_expires_after_max_age(self):
        t = IoUTracker(max_age=2)
        (tid,) = t.update([(0, 0, 50, 50)])
        for _ in range(3):
            t.update([])
        (new_tid,) = t.update([(0, 0, 50, 50)])
        self.assertNotEqual(tid, new_tid)


if __name__ == "__main__":
    unittest.main()
