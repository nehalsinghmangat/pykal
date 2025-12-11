"""
Control algorithms for pykal.

This module provides implementations of various control algorithms
including PID controllers, LQR, MPC, and other control strategies.
"""

from .pid import PID as pid

__all__ = ["pid", "PID"]
