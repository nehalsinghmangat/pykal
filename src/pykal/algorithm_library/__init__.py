"""
pykal.algorithm_library

Collection of implemented control and estimation algorithms.

This module provides access to peer-reviewed algorithms from the robotics
and control theory literature, implemented as pure Python functions compatible
with pykal's DynamicalSystem abstraction.

Submodules:
    estimators: State estimation algorithms (Kalman filters, observers, etc.)
    controllers: Control algorithms (PID, LQR, MPC, etc.)
"""

from . import estimators
from . import controllers

__all__ = ["estimators", "controllers"]
