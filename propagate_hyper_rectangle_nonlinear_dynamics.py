#!/usr/bin/env python
"""
Created on Fri Jul  6 22:29:01 2018

@author: gowtham
"""
from propagate_hyper_rectangle_linear_dynamics import (HyperRectangle,
                                                       propagateRectangle)
from optimal_control_framework.dynamics import AbstractDynamicSystem




# Procedure
# Given previous hyper rectangles, nonlinear dynamics
# First linearize around mean
# Propagate using linearized dynamics (Get both rectangle and input points)
# Then formulate n optimizations to find the scale for nonlinear dynamics
# Send back the hyper rectangle

# Also formulate a linearized version with closed form solution instead of
# full optimization