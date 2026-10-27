import unittest

from zone_exit.geometry import box_center, denormalize, iou, point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
L_SHAPE = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]


class GeometryTests(unittest.TestCase):
    def test_point_inside_and_outside_square(self):
        self.assertTrue(point_in_polygon(5, 5, SQUARE))
        self.assertFalse(point_in_polygon(15, 5, SQUARE))
        self.assertFalse(point_in_polygon(5, -1, SQUARE))

    def test_concave_polygon(self):
        self.assertTrue(point_in_polygon(2, 8, L_SHAPE))    # in the vertical arm
        self.assertFalse(point_in_polygon(8, 8, L_SHAPE))   # in the notch

    def test_denormalize(self):
        self.assertEqual(denormalize([(0.5, 0.25)], 200, 100), [(100.0, 25.0)])

    def test_box_center(self):
        self.assertEqual(box_center((0, 0, 10, 20)), (5.0, 10.0))

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=5)


if __name__ == "__main__":
    unittest.main()
