from typing import (
    Callable,
    Optional,
)
from pykal_core.control_system.system import System
import numpy as np
from numpy.typing import NDArray


class PID:
    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 0.0,
        umin: Optional[float] = None,
        umax: Optional[float] = None,
    ) -> None:
        """
        Initialize a PID controller.

        Parameters
        ----------
        Kp, Ki, Kd : float
            Proportional, integral, and derivative gains.
        umin, umax : float, optional
            Optional saturation limits on the control output.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.umin = umin
        self.umax = umax

        self.integral_error = None  # type: Optional[NDArray]
        self.prev_error = None  # type: Optional[NDArray]

    def reset(self):
        """Reset integral and derivative history."""
        self.integral_error = None
        self.prev_error = None

    def compute(
        self,
        x_est: NDArray,  # state estimate (n, 1)
        x_ref: NDArray,  # reference signal (n, 1)
        dt: float,  # timestep
    ) -> NDArray:
        """
        Compute the PID control signal.

        Returns
        -------
        NDArray
            Control signal (m, 1), aligned with input dimension of self.sys.
        """
        error = x_ref - x_est  # (n, 1)

        # Initialize memory if needed
        if self.integral_error is None:
            self.integral_error = np.zeros_like(error)
        if self.prev_error is None:
            self.prev_error = np.zeros_like(error)

        # PID terms
        P = self.Kp * error
        self.integral_error += error * dt
        I = self.Ki * self.integral_error
        D = self.Kd * (error - self.prev_error) / dt

        self.prev_error = error

        u = P + I + D  # (n, 1)

        # Saturation
        if self.umin is not None or self.umax is not None:
            u = np.clip(u, self.umin, self.umax)

        return u


class Controller:
    def __init__(self, sys: System):
        self.sys = sys
        self.pid = PID()
