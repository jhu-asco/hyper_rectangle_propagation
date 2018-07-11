# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:14:50 2018

@author: gowtham
"""

from optimal_control_framework.controllers import AbstractController
import numpy as np


class LinearController(AbstractController):
    def __init__(self, dynamics, K, k):
        super(LinearController, self).__init__(dynamics)
        self.K = K
        self.k = k

    def jacobian(self, i, x):
        return self.K

    def deterministic_control(self, i, x):
        return np.dot(self.K, x) + self.k
