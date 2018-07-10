#!/usr/bin/env python
"""
Created on Fri Jul  6 13:57:19 2018

@author: gowtham
"""

import numpy as np
from hyper_rectangle import HyperRectangle


def scale(V):
    sign_V = np.sign(V)
    D = np.sum(V * sign_V, axis=1)
    return D, sign_V


def is_equal(array, value, tol=1e-8):
    if np.all(np.abs(np.abs(array) - value) < tol):
        return True
    return False


def propagateSamples(x0_rectangle, w_rectangle, linear_dynamics_params,
                     N, Nsamples):
    np.random.seed(123343)
    A, G = linear_dynamics_params
    n = A.shape[0]
    RS0 = x0_rectangle.R * x0_rectangle.S
    RS_w = w_rectangle.R * w_rectangle.S
    xs = np.empty((Nsamples, N + 1, n))
    for i in range(Nsamples):
        e0 = 2 * np.random.sample(n) - 1
        xs[i, 0] = x0_rectangle.mu + np.dot(RS0, e0)
        for j in range(N):
            e_w = np.random.sample(n) * 2 - 1
            w = w_rectangle.mu + np.dot(RS_w, e_w)
            xs[i, j + 1] = np.dot(A, xs[i, j]) + np.dot(G, w)
    return xs


def propagateRectangle(x_rectangle, w_rectangle, linear_dynamics_params):
    # Fitting hyper rectangle at next time step
    A, G = linear_dynamics_params
    mu_i, R_i, S_i = x_rectangle
    _, R_w_i, S_w_i = w_rectangle
    M = np.hstack((np.dot(A, R_i * S_i), np.dot(G, R_w_i * S_w_i)))
    U, S, V = np.linalg.svd(M, full_matrices=False)
    diag_U = np.diag(U)
    if is_equal(diag_U, 1) or is_equal(diag_U, 0):
        U = R_i
        S = np.ones_like(S)
        V = np.dot(R_i.T, M)
    D, e_in = scale(V)
    S_next = S * D
    mu_next = np.dot(A, mu_i)
    # input_points[:, 0] corresponds to input point for 1st axis
    return HyperRectangle(mu_next, U, S_next), e_in.T
