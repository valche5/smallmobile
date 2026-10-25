import unittest

from probe import choose, estimate, world


class ProbeTests(unittest.TestCase):
    def test_holdout_can_reverse_the_selection_gain(self):
        selection = {(0, 1): 0, (0, 2): 1, (1, 1): 0, (1, 2): 1}
        self.assertEqual(choose(1, [1, 2], selection), 2)
        # Le score ne tronque ni les régressions ni les positions où le témoin est retenu.
        score, _ = estimate({(2, "late"): [-0.5, 0.0]}, 200)
        self.assertEqual(score, -0.25)

    def test_equal_weight_for_each_cell(self):
        score, _ = estimate({(2, "early"): [0.1] * 10, (4, "late"): [-0.3] * 2}, 200)
        self.assertAlmostEqual(score, -0.1)

    def test_one_position_is_not_hundreds_of_independent_observations(self):
        _, ci = estimate({(2, "late"): [0.5]}, 200)
        self.assertIsNone(ci)

    def test_confirmation_seeds_are_separate(self):
        a = {world(51001, 3, "select", i) for i in range(256)}
        b = {world(51001, 3, "confirm", i) for i in range(256)}
        self.assertFalse(a & b)


if __name__ == "__main__":
    unittest.main()
