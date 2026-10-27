import unittest

from zone_exit.exit_logic import ExitRule, ZoneExitMonitor

ZONE = [(100, 0), (300, 0), (300, 100), (100, 100)]


def box_at(cx, cy=50, half=10):
    return (cx - half, cy - half, cx + half, cy + half)


def run_path(monitor, xs, dt=0.1, tid=1):
    events = []
    for i, x in enumerate(xs):
        events += monitor.update({tid: box_at(x)}, now=i * dt)
    return events


class ExitLogicTests(unittest.TestCase):
    def rule(self, **kw):
        base = dict(direction="left_to_right", min_distance_px=40, min_time_in_zone_s=0.5)
        base.update(kw)
        return ExitRule(**base)

    def test_triggers_once_when_moving_in_allowed_direction(self):
        m = ZoneExitMonitor(ZONE, self.rule())
        events = run_path(m, range(110, 250, 10))
        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(events[0].dx, 40)

    def test_wrong_direction_does_not_trigger(self):
        m = ZoneExitMonitor(ZONE, self.rule())
        self.assertEqual(run_path(m, range(250, 110, -10)), [])

    def test_too_short_time_does_not_trigger(self):
        m = ZoneExitMonitor(ZONE, self.rule(min_time_in_zone_s=5.0))
        self.assertEqual(run_path(m, range(110, 250, 10)), [])

    def test_small_movement_does_not_trigger(self):
        m = ZoneExitMonitor(ZONE, self.rule())
        self.assertEqual(run_path(m, [150, 151, 152, 150, 149] * 5), [])

    def test_vehicle_outside_zone_is_ignored(self):
        m = ZoneExitMonitor(ZONE, self.rule())
        self.assertEqual(run_path(m, range(10, 90, 10)), [])

    def test_any_direction(self):
        m = ZoneExitMonitor(ZONE, self.rule(direction="any"))
        self.assertEqual(len(run_path(m, range(250, 110, -10))), 1)

    def test_two_vehicles_are_independent(self):
        m = ZoneExitMonitor(ZONE, self.rule())
        events = []
        for i, x in enumerate(range(110, 250, 10)):
            events += m.update({1: box_at(x), 2: box_at(290 - (x - 110))}, now=i * 0.1)
        self.assertEqual([e.track_id for e in events], [1])

    def test_stale_tracks_are_dropped(self):
        m = ZoneExitMonitor(ZONE, self.rule(stale_after_s=1.0))
        m.update({1: box_at(150)}, now=0.0)
        m.update({}, now=5.0)
        self.assertEqual(m._states, {})

    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            ExitRule(direction="sideways")

    def test_zone_needs_three_points(self):
        with self.assertRaises(ValueError):
            ZoneExitMonitor([(0, 0), (1, 1)], ExitRule())


if __name__ == "__main__":
    unittest.main()
