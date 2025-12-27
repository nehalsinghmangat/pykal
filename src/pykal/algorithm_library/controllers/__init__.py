"""
pykal.algorithm_library.controllers

Control algorithms from the robotics and control theory literature.

Available Modules:
    pid: PID controller and variants (standard, anti-windup, etc.)
    lqr: Linear Quadratic Regulator for optimal state feedback control
    mpc: Model Predictive Control for constrained optimal control
"""

from . import pid
from . import lqr
from . import mpc

__all__ = ["pid", "lqr", "mpc"]
