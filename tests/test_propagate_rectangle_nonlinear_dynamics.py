#!/usr/bin/env python2
import hyper_rectangle_propagation.propagate_hyper_rectangle_nonlinear_dynamics as PropagateRect
from hyper_rectangle_propagation.propagate_hyper_rectangle_linear_dynamics import (
    rotation, HyperRectangle, propagateRectangle)
from optimal_control_framework.dynamics import AbstractDynamicSystem
from optimal_control_framework.controllers import AbstractController
import unittest
import numpy.testing as np_testing
import numpy as np


class LinearDynamics(AbstractDynamicSystem):
    def __init__(self, theta):
        self.A = theta[0]
        self.B = theta[1]
        self.G = theta[2]
        self.n = self.A.shape[0]
        if len(self.B.shape) == 2:
            self.m = self.B.shape[1]
        else:
            self.m = 1

    def jacobian(self, t, x, u, wbar):
        return [self.A, self.B, self.G]

    def xdot(self, t, x, u, w):
        return np.dot(self.A, x) + np.dot(self.B, u) + np.dot(self.G, w)


class LinearController(AbstractController):
    def __init__(self, dynamics, K):
        self.K = K
        super(LinearController, self).__init__(dynamics)

    def jacobian(self, i, x):
        return self.K

    def deterministic_control(self, i, x):
        return np.dot(self.K, x)


class TestFitHyperRectangle(unittest.TestCase):
    def testFitHyperRectangleLinearDyn(self):
        # Linear params
        A = np.array([[0, 1], [0, 0]])
        #A = np.zeros(2)
        B = np.array([0, 1])
        G = np.eye(2)
        K = np.array([-1, -1])
        #K = np.array([0, 0])
        linear_dynamics = LinearDynamics((A, B, G))
        linear_controller = LinearController(linear_dynamics, K)

        # State hyper rectangle params
        mu_i = np.array([1, 0])
        theta_i = np.pi / 4
        R_i = rotation(theta_i)
        S_i = np.array([1, 1])
        # Noise hyper rectangle params
        theta_w_i = np.pi / 4
        R_w_i = rotation(theta_w_i)
        S_w_i = np.array([0.2, 0.4])
        rect_x = HyperRectangle(mu_i, R_i, S_i)
        rect_w = HyperRectangle(np.zeros(2), R_w_i, S_w_i)
        # Propagate
        dt = 1.0
        rect_out, in_points = PropagateRect.propagateRectangle(
        0, dt, rect_x, rect_w, linear_dynamics, linear_controller)
        Abar = np.eye(2) + dt * (A + np.outer(B, K))
        Gbar = G * dt
        l_rect_out, l_in_points = propagateRectangle(rect_x, rect_w,
                                                     (Abar, Gbar))
        np_testing.assert_almost_equal(rect_out.mu, l_rect_out.mu)
        np_testing.assert_almost_equal(rect_out.R, l_rect_out.R)
        np_testing.assert_almost_equal(rect_out.S, l_rect_out.S)
        np_testing.assert_almost_equal(in_points, l_in_points)
#        np_testing.assert_almost_equal(rect_out.mu, mu_i)
#        np_testing.assert_almost_equal(rect_out.R[:, 0], -1*R_i[:, 0])
#        np_testing.assert_almost_equal(rect_out.R[:, 1], R_i[:, 1])
#        np_testing.assert_almost_equal(rect_out.S, S_i + S_w_i)
#        in_points[:, 0] = -1*in_points[:, 0]
#        np_testing.assert_almost_equal(in_points,
#                                       np.vstack((np.eye(2), np.eye(2))))

    def testFitHyperRectangleLinearizedDyn(self):
        # Linear params
        A = np.array([[0, 1], [0, 0]])
        #A = np.zeros(2)
        B = np.array([0, 1])
        G = np.eye(2)
        K = np.array([-1, -1])
        #K = np.array([0, 0])
        linear_dynamics = LinearDynamics((A, B, G))
        linear_controller = LinearController(linear_dynamics, K)

        # State hyper rectangle params
        mu_i = np.array([1, 0])
        theta_i = np.pi / 4
        R_i = rotation(theta_i)
        S_i = np.array([1, 1])
        # Noise hyper rectangle params
        theta_w_i = np.pi / 4
        R_w_i = rotation(theta_w_i)
        S_w_i = np.array([0.2, 0.4])
        rect_x = HyperRectangle(mu_i, R_i, S_i)
        rect_w = HyperRectangle(np.zeros(2), R_w_i, S_w_i)
        # Propagate
        dt = 1.0
        rect_out, in_points = PropagateRect.propagateLinearizedRectangle(
        0, dt, rect_x, rect_w, linear_dynamics, linear_controller, 1e-2)
        Abar = np.eye(2) + dt * (A + np.outer(B, K))
        Gbar = G * dt
        l_rect_out, l_in_points = propagateRectangle(rect_x, rect_w,
                                                     (Abar, Gbar))
        np_testing.assert_almost_equal(rect_out.mu, l_rect_out.mu)
        np_testing.assert_almost_equal(rect_out.R, l_rect_out.R)
        np_testing.assert_almost_equal(rect_out.S, l_rect_out.S)
        np_testing.assert_almost_equal(in_points, l_in_points)


if __name__ == "__main__":
    unittest.main()
