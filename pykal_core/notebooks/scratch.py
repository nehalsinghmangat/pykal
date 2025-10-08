from pykal_core.system import System
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Callable


# Radar and time settings
d = 100000
h0 = 100000
DT = 0.005
measurement_period = 1.0
Tf = 30.0
N = int(Tf / DT) + 1
t = np.linspace(0, Tf, N)

Gk = np.eye(3)

# Constants
po = 2  # lb sec^2 / ft^4
g = 32.2  # ft/s^2
kp = 20000  # ft

# Initial truth state: position, velocity, ballistic coeff
x0 = np.array([300000, -20000, 1e-3], dtype=np.float64)

# Initial uncertainty scaling factor
factor = 1.2
s = np.array([30000 * factor, 2000 * factor, 1.4e-3 * factor])
Pk = np.diag(s**2)
P0 = Pk.copy()


def Q_fall() -> NDArray:
    return np.zeros((3, 3))


def R_radar() -> NDArray:
    return np.array([[10000]])


# Linearized measurement Jacobian
def Hk_lin(x: NDArray):
    denom = np.sqrt(d**2 + (x[0] - h0) ** 2)
    return np.array([[(x[0] - h0) / denom, 0, 0]])


# Measurement function
def meas(x: NDArray) -> NDArray:
    return np.array([np.sqrt(d**2 + (x[0] - h0) ** 2)])


# Dynamics function (true continuous)
def true_dynamics(t: float, x: NDArray) -> NDArray:
    dx1 = x[1]
    dx2 = np.exp(-x[0] / kp) * (x[1] ** 2) * x[2] - g
    dx3 = 0.0
    return np.array([dx1, dx2, dx3])


# Discrete dynamics for propagation
def discrete_dyn(tk: float, xk: NDArray) -> NDArray:
    return xk + DT * true_dynamics(tk, xk)


# Linearized F matrix
def F_linearized_discrete(x):
    F = np.eye(3)
    F[1, 0] = (x[1] ** 2 * x[2]) * np.exp(-x[0] / kp) * (-1 / kp)
    F[1, 1] = 2 * np.exp(-x[0] / kp) * x[1] * x[2]
    F[1, 2] = np.exp(-x[0] / kp) * x[1] ** 2
    F *= DT
    return np.eye(3) + F


sys = System(
    f=discrete_dyn,
    state_names=["altitude", "velocity", "ballistic_coeff"],
    h=meas,
    measurement_names=["slant_range"],
    system_type="dti",
    R=R_radar,
    Q=Q_fall,
)
