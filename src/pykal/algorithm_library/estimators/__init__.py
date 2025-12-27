"""
pykal.algorithm_library.estimators

State estimation algorithms from the robotics and control theory literature.

Available Modules:
    kf: Kalman filter and Extended Kalman Filter (EKF)
    ukf: Unscented Kalman Filter (UKF)
    pf: Particle Filter for nonlinear/non-Gaussian estimation
    ai_ukf: Augmented Information UKF for hybrid physics-data-driven estimation
"""

from . import kf
from . import ukf
from . import pf
from . import ai_ukf

__all__ = ["kf", "ukf", "pf", "ai_ukf"]
