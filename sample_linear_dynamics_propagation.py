#!/usr/bin/env python
"""
Created on Tue Jul 10 14:03:21 2018

@author: gowtham
"""
from rotation2d import rotation
from hyper_rectangle import HyperRectangle
from propagate_hyper_rectangle_linear_dynamics import (propagateRectangle,
                                                       propagateSamples)
from plot_2D_hyper_rectangles import (plot2DHyperRectangle, plot2DSamples)
import numpy as np
import matplotlib.pyplot as plt

# For now considering a 2D system easy to visualize
# Linear params
A = np.diag([0.9, 0.9])
A[0, 1] = 0.2
G = np.diag([0.1, 0.1])
dynamic_params = (A, G)
# State hyper rectangle params
mu_i = np.array([1, 1])
theta_i = 0
R_i = rotation(theta_i)
S_i = np.array([0.1, 0.2])

# Noise hyper rectangle params
theta_w_i = 0.0
R_w_i = rotation(theta_w_i)
S_w_i = np.array([0.01, 0.011])

# Trajectory parameters
N = 100
rect_x0 = HyperRectangle(mu_i, R_i, S_i)
rect_w = HyperRectangle(np.zeros_like(mu_i), R_w_i, S_w_i)
mu_array = np.empty((N + 1, 2))
# Display
f = plt.figure(2)
plt.clf()
rect_x = rect_x0
ax = f.add_subplot(111, aspect='equal')
plot2DHyperRectangle(rect_x, ax=ax)
mu_array[0] = rect_x.mu

for i in range(N):
    rect_out, _ = propagateRectangle(rect_x, rect_w, dynamic_params)
    rect_x = rect_out
    mu_array[i + 1] = rect_x.mu
    plot2DHyperRectangle(rect_x, ax=ax)
ax.plot(mu_array[:, 0], mu_array[:, 1], 'bo-', linewidth=2)
# Plot samples
Nsamples = 1000
xsamples = propagateSamples(rect_x0, rect_w, dynamic_params, N, Nsamples)
plot2DSamples(xsamples, ax)
print(rect_out)
