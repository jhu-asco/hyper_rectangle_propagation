#!/usr/bin/env python2
from fit_hyper_rectangles.propagate_hyper_rectangle_linear_dynamics import *
import unittest
import numpy.testing as np_testing
import numpy as np


class TestFitHyperRectangle(unittest.TestCase):
    def testFitHyperRectangle(self):
        # Linear params
        A = np.eye(2)
        G = np.eye(2)
        # State hyper rectangle params
        mu_i = np.array([1, 0])
        theta_i = np.pi / 4
        R_i = rotation(theta_i)
        S_i = np.array([1, 1])
        # Noise hyper rectangle params
        theta_w_i = np.pi / 4
        R_w_i = rotation(theta_w_i)
        S_w_i = np.array([2.5, 2])
        rect_x = HyperRectangle(mu_i, R_i, S_i)
        rect_w = HyperRectangle(np.zeros(2), R_w_i, S_w_i)
        rect_out, in_points = propagateRectangle(rect_x, rect_w, (A, G))
        np_testing.assert_almost_equal(rect_out.mu, mu_i)
        np_testing.assert_almost_equal(rect_out.R[:, 0], -1 * R_i[:, 0])
        np_testing.assert_almost_equal(rect_out.R[:, 1], R_i[:, 1])
        np_testing.assert_almost_equal(rect_out.S, S_i + S_w_i)
        in_points[:, 0] = -1 * in_points[:, 0]
        np_testing.assert_almost_equal(in_points,
                                       np.vstack((np.eye(2), np.eye(2))))


if __name__ == "__main__":
    unittest.main()
