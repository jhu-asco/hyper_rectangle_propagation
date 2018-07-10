#!/usr/bin/env python

import numpy as np


def rotation(theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    return np.array([[c_theta, -s_theta], [s_theta, c_theta]])
