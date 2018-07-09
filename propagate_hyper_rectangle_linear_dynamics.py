#!/usr/bin/env python
"""
Created on Fri Jul  6 13:57:19 2018

@author: gowtham
"""

import numpy as np
from collections import namedtuple

HyperRectangle = namedtuple('HyperRectangle', 'mu, R, S')


def rotation(theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    return np.array([[c_theta, -s_theta], [s_theta, c_theta]])


def findMax(v, tol=1e-3):
    n = int(0.5 * v.size)
    v1 = v[:n]
    v2 = v[n:]
    max_v1 = np.max(np.abs(v1))
    max_v2 = np.max(np.abs(v2))
    max_value = 0
    if max_v1 > tol:
        max_value = max_value + np.dot(v1, v1) / max_v1
    if max_v2 > tol:
        max_value = max_value + np.dot(v2, v2) / max_v2
    return max_value


def scale(V, tol=1e-6):
    max_V = np.max(np.abs(V), axis=0)
    np.clip(max_V, tol, None, out=max_V)
    scaled_V = V / max_V
    D = np.sum(V * scaled_V, axis=0)
    return D, scaled_V


def propagateRectangle(x_rectangle, w_rectangle, linear_dynamics_params):
    # Fitting hyper rectangle at next time step
    A, G = linear_dynamics_params
    mu_i, R_i, S_i = x_rectangle
    _, R_w_i, S_w_i = w_rectangle
    M = np.hstack((np.dot(A, R_i * S_i), np.dot(G, R_w_i * S_w_i)))
    U, S, V = np.linalg.svd(M, full_matrices=False)
    n = S.size
    D1, ei_in = scale(V[:, :n].T)
    D2, ewi_in = scale(V[:, n:].T)
    S_next = S * (D1 + D2)
    mu_next = np.dot(A, mu_i)
    input_points = np.vstack((ei_in, ewi_in))
    # input_points[:, 0] corresponds to input point for 1st axis
    return HyperRectangle(mu_next, U, S_next), input_points


if __name__ == "__main__":
    # For now considering a 2D system easy to visualize
    # Linear params
    A = np.eye(2)
    #A[0, 1] = 0.01
    G = np.eye(2)
    # G = 0.1*np.diag([1, 0.5])
    # State hyper rectangle params
    mu_i = np.array([1, 0])
    theta_i = np.pi / 4
    R_i = rotation(theta_i)
    S_i = np.array([1, 1])
    # Noise hyper rectangle params
    theta_w_i = np.pi / 4
    R_w_i = rotation(theta_w_i)
    S_w_i = np.array([2, 2.5])
    rect_out, _ = propagateRectangle(HyperRectangle(mu_i, R_i, S_i),
                                  HyperRectangle(None, R_w_i, S_w_i),
                                  (A, G))
    print(rect_out)
    # 3D test case
    A = np.random.sample((3, 3))
    G = np.random.sample((3, 3))

    mu_i = np.array([1, 1, 1])
    R_i = np.eye(3)
    S_i = np.array([2, 2, 2])
    R_w_i = np.array([R_i[1], R_i[0], R_i[2]])
    S_w_i = np.array([0.1, 0.1, 0.1])
    rect_out2, _ = propagateRectangle(HyperRectangle(mu_i, R_i, S_i),
                                   HyperRectangle(None, R_w_i, S_w_i),
                                   (A, G))
    print(rect_out2)
