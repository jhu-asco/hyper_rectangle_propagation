#!/usr/bin/env python
"""
Created on Fri Jul  6 22:29:01 2018

@author: gowtham
"""
import numpy as np
import hyper_rectangle_propagation.propagate_hyper_rectangle_linear_dynamics as PropLinearRect
from scipy.optimize import least_squares
from hyper_rectangle import HyperRectangle


def projectToRectangle(rectangle, e):
    RS = rectangle.R * rectangle.S
    return rectangle.mu + np.dot(RS, e)


def axis_residual(e, mun, max_r, rvec, i, dt, x_rectangle,
                  w_rectangle, dynamics, controller):
    n = dynamics.n
    x_in = projectToRectangle(x_rectangle, e[:n])
    w_in = projectToRectangle(w_rectangle, e[n:])
    u_in = controller.deterministic_control(i, x_in)
    x_out = x_in + dt * dynamics.xdot(i, x_in, u_in, w_in)
    x_diff = x_out - mun
    cost_res = (max_r - np.abs(np.dot(x_diff, rvec)))
    return cost_res


def getConstraintDirection(e, tol=1e-6):
    i = np.argmax(np.abs(e))
    sign = np.sign(e[i])
    out = np.zeros_like(e)
    if np.abs(e[i] - 1) < tol:
        out[i] = sign * 1
    return out


def projectVector(vin, e, tol=1e-6):
    sign_vin = np.sign(vin)
    idx = np.abs(sign_vin * e - 1) < tol
    vin[idx] = 0


def scaleDirectionVector(delta, step_size, tol=1e-6):
    current_size = max(np.linalg.norm(delta), tol)
    return (step_size / current_size) * delta


def getClosedLoopDynamicMatrix(dt, A, B, K):
    if len(B.shape) == 1:
        BK = np.outer(B, K)
    else:
        BK = np.dot(B, K)
    Abar = np.eye(A.shape[0]) + dt * (A + BK)
    return Abar


def linearizedUpdate(e, rvec, step_size, mu_next, i, dt, x_rectangle,
                     w_rectangle, dynamics, controller):
    n = dynamics.n
    x_in = projectToRectangle(x_rectangle, e[:n])
    w_in = projectToRectangle(w_rectangle, e[n:])
    u_in = controller.deterministic_control(i, x_in)
    K = controller.jacobian(i, x_in)
    A, B, G = dynamics.jacobian(i, x_in, u_in, w_in)
    Abar = getClosedLoopDynamicMatrix(dt, A, B, K)

    Abar_Rx_S = np.dot(Abar, x_rectangle.R) * x_rectangle.S
    delta_e_dirxn = np.dot(Abar_Rx_S.T, rvec)
    projectVector(delta_e_dirxn, e[:n])
    delta_e = scaleDirectionVector(delta_e_dirxn, step_size)
    e_x_new = e[:n] + delta_e
    np.clip(e_x_new, -1, 1, out=e_x_new)

    G_Rw_S = np.dot(G, w_rectangle.R) * w_rectangle.S
    delta_e_w_dirxn = np.dot(G_Rw_S.T, rvec)
    projectVector(delta_e_w_dirxn, e[n:])
    delta_e_w = scaleDirectionVector(delta_e_w_dirxn, step_size)
    e_w_new = e[n:] + delta_e_w
    np.clip(e_w_new, -1, 1, out=e_w_new)

    # Compute outputs
    x_in_up = projectToRectangle(x_rectangle, e_x_new)
    w_in_up = projectToRectangle(w_rectangle, e_w_new)
    u_in_up = controller.control(i, x_in_up)
    x_out_up = x_in_up + dt * dynamics.xdot(i, x_in_up, u_in_up, w_in_up)
    scale = np.abs(np.dot(rvec, (x_out_up - mu_next)))
    e_new = np.hstack((e_x_new, e_w_new))
    return scale, e_new


def propagateMean(i, dt, x_rectangle, w_rectangle, dynamics, controller):
    # We assume the w_rectangle.mu = 0 always!!
    n = dynamics.n
    u0 = controller.deterministic_control(i, x_rectangle.mu)
    A, B, G = dynamics.jacobian(i, x_rectangle.mu, u0, w_rectangle.mu)
    K = controller.jacobian(i, x_rectangle.mu)
    Abar = getClosedLoopDynamicMatrix(dt, A, B, K)
    x_mod_rectangle = HyperRectangle(np.zeros(n), x_rectangle.R,
        x_rectangle.S)
    out_rect, in_points = PropLinearRect.propagateRectangle(
        x_mod_rectangle, w_rectangle, (Abar, G * dt))
    mun = x_rectangle.mu + dt * \
        dynamics.xdot(i, x_rectangle.mu, u0, np.zeros(n))
    return mun, out_rect, in_points


def propagateRectangle(i, dt, x_rectangle, w_rectangle, dynamics, controller):
    # We assume the w_rectangle.mu = 0 always!!
    n = dynamics.n
    mun, out_rect, in_points = propagateMean(i, dt, x_rectangle, w_rectangle,
                                             dynamics, controller)
    Rn = out_rect.R
    # lb = -1*np.ones(2*n)
    # ub = np.ones(2*n)
    max_r = 10 * np.max(out_rect.S)
    scale_out = []
    input_points = []
    for i in range(n):
        args = (mun, max_r, Rn[:, i], i, dt, x_rectangle, w_rectangle,
                dynamics, controller)
        res = least_squares(axis_residual, in_points[:, i], bounds=(-1, 1),
                            args=args,
                            max_nfev=1000)
        opt_e = res.x
        # opt_e = in_points[:, i]
        opt_res = axis_residual(opt_e, *args)
        scale_out.append(max_r - opt_res)
        input_points.append(opt_e)
    return (HyperRectangle(mun, Rn, np.array(scale_out)),
            np.vstack(input_points).T)


def propagateLinearizedRectangle(i, dt, x_rectangle, w_rectangle, dynamics,
                                 controller, step_size):
    # We assume the w_rectangle.mu = 0 always!!
    n = dynamics.n
    mun, out_rect, in_points = propagateMean(i, dt, x_rectangle, w_rectangle,
                                             dynamics, controller)
    Rn = out_rect.R
    scale_out = []
    input_points = []
    for i in range(n):
        scale, e_new = linearizedUpdate(
            in_points[:, i], Rn[:, i], step_size, mun,
            i, dt, x_rectangle, w_rectangle, dynamics, controller)
        scale_out.append(scale)
        input_points.append(e_new)
    return (HyperRectangle(mun, Rn, np.array(scale_out)),
            np.vstack(input_points).T)
