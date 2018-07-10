#!/usr/bin/env python

from hyper_rectangle import HyperRectangle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def projectRectangle(hyper_rectangle):
    n = hyper_rectangle.mu.size
    if n == 2:
        return hyper_rectangle
    elif n < 2:
        raise RuntimeError("Cannot project a rectangle smaller"
                           " than 2 dimensions")
    mu_proj = hyper_rectangle.mu[:2]
    RSproj = hyper_rectangle.R[:2, :] * hyper_rectangle.S
    U, S, V = np.linalg.svd(RSproj, full_matrices=False)
    R_proj = U
    S_proj = S * (np.sum(np.abs(V), axis=1))
    return HyperRectangle(mu_proj, R_proj, S_proj)


def detR(R):
    return R[0, 0] * R[1, 1] - R[1, 0] * R[0, 1]


def plot2DSamples(xs, ax):
    Ns, N, n = xs.shape
    assert(n == 2)
    for i in range(Ns):
        ax.plot(xs[i, :, 0], xs[i, :, 1], 'ro-', linewidth=0.5)


def plot2DHyperRectangle(hyper_rectangle, ax=None, fig=None):
    if ax is None:
        fig = plt.figure(fig)
        ax = fig.add_subplot(111, aspect='equal')

    rect_proj = projectRectangle(hyper_rectangle)
    e_bottom_left = np.array([-1, -1])
    if detR(rect_proj.R) < 0:
        Rproj = np.hstack([rect_proj.R[:, [1]], rect_proj.R[:, [0]]])
        S = np.array([rect_proj.S[1], rect_proj.S[0]])
    else:
        Rproj = rect_proj.R
        S = rect_proj.S
    bottom_left = rect_proj.mu + np.dot(Rproj * S, e_bottom_left)
    angle = np.arctan2(Rproj[1, 0],
                       Rproj[0, 0]) * 180.0 / np.pi
    rect_patch = patches.Rectangle(bottom_left, 2 * S[0],
                                   2 * S[1], angle,
                                   fill=False)
    ax.add_patch(rect_patch)
