#!/usr/bin/env python
"""
Created on Tue Jul 10 14:03:21 2018

@author: gowtham
"""
from hyper_rectangle import HyperRectangle
from propagate_hyper_rectangle_nonlinear_dynamics import (
    propagateRectangle, propagateLinearizedRectangle)
from plot_2D_hyper_rectangles import (plot2DHyperRectangle, plot2DSamples)
from optimal_control_framework.dynamics import UnicycleDynamics
from linear_controller import LinearController
import numpy as np
import matplotlib.pyplot as plt

# Unicycle system
unicycle_dynamics = UnicycleDynamics()
# Controller:
K = np.zeros((2, 3))
k = np.array([0.5, 0.])
controller = LinearController(unicycle_dynamics, K, k)
# State hyper rectangle params
mu_i = np.array([0, 0, 0])
R_i = np.eye(3)
S_i = np.array([0.1, 0.1, 0.1]) * 0.1

# Noise hyper rectangle params
R_w_i = np.eye(3)
S_w_i = 0 * np.array([0.01, 0.01, 0.01])  # noise in vx, vy, thetadot

# Trajectory parameters
N = 50
dt = 0.05
rect_x0 = HyperRectangle(mu_i, R_i, S_i)
rect_w = HyperRectangle(np.zeros_like(mu_i), R_w_i, S_w_i)
mu_array = np.empty((N + 1, 3))
# Display
f = plt.figure(2)
plt.clf()
ax = f.add_subplot(111, aspect='equal')
plot2DHyperRectangle(rect_x0, ax=ax)
mu_array[0] = rect_x0.mu

# Propagate
rect_x = rect_x0
for i in range(N):
    rect_out, _ = propagateRectangle(i, dt, rect_x, rect_w, unicycle_dynamics,
                                     controller)
    rect_x = rect_out
    mu_array[i + 1] = rect_x.mu
    plot2DHyperRectangle(rect_x, ax=ax)
ax.plot(mu_array[:, 0], mu_array[:, 1], 'bo-', linewidth=2)
# Plot samples
#Nsamples = 1000
#xsamples = propagateSamples(rect_x0, rect_w, dynamic_params, N, Nsamples)
#plot2DSamples(xsamples, ax)
# print(rect_out)
