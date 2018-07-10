#!/usr/bin/env python

import unittest
import numpy as np
import numpy.testing as np_testing
from hyper_rectangle_propagation.hyper_rectangle import HyperRectangle
from hyper_rectangle_propagation.rotation2d import rotation


class TestHyperRectangle(unittest.TestCase):
    def testProject2DRectangle(self):
        in_rect = HyperRectangle(
            np.zeros(2), rotation(np.pi / 3), np.array([1, 2]))
        out_rect = projectRectangle(in_rect)
        np_testing.assert_allclose(in_rect.mu, out_rect.mu)
        np_testing.assert_allclose(in_rect.R, out_rect.R)
        np_testing.assert_allclose(in_rect.S, out_rect.S)

    def testProjectUnrotated3DRectangle(self):
        in_rect = HyperRectangle(np.zeros(3), np.eye(3), np.array([3, 2, 1]))
        out_rect = projectRectangle(in_rect)
        np_testing.assert_allclose(out_rect.mu, in_rect.mu[:2])
        np_testing.assert_allclose(out_rect.R, np.eye(2))
        np_testing.assert_allclose(out_rect.S, in_rect.S[:2])

    def testProjectRotated3DRectangle(self):
        Rin = np.eye(3)
        Rin[1:3, 1:3] = rotation(np.pi / 4)
        in_rect = HyperRectangle(np.zeros(3), Rin, np.array([3, 1, 1]))
        out_rect = projectRectangle(in_rect)
        pred_S = np.array([3, np.sqrt(2)])
        np_testing.assert_allclose(out_rect.mu, in_rect.mu[:2])
        np_testing.assert_allclose(out_rect.R, np.eye(2))
        np_testing.assert_allclose(out_rect.S, pred_S)


if __name__ == "__main__":
    unittest.main()
