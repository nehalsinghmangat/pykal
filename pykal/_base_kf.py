from typing import Callable, Optional
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from system import System


class BaseKF(ABC):
    """
    Abstract base class for Kalman Filter estimators in the `pykal` framework.

    This class provides a standardized interface for implementing Kalman filtering algorithms.
    All subclasses must implement the `predict`, `update`, and `run` methods.

    Parameters
    ----------
    sys : System
        An instance of the `System` class, which encapsulates the system dynamics (`f`),
        measurement model (`h`), and optional noise models (`Q`, `R`). The system must also
        define the timing structure (`SystemType`) and variable name metadata.

    Requirements
    ------------
    The provided `System` object must conform to strict function input/output standards:
      - `f(x, u, t)` returns the time derivative or next state (n×1 ndarray)
      - `h(x, u, t)` returns measurements (m×1 ndarray)
      - `Q(x, u, t)` returns process noise covariance (n×n ndarray)
      - `R(x, u, t)` returns measurement noise covariance (m×m ndarray)

    All callable models are validated using `validate_system_function` from `utils.iosafety`
    and are accessed safely using `call_validated_function_with_args`.

    Notes
    -----
    The `BaseKF` class does not implement filtering logic directly. It only defines
    the contract that derived filters must follow. This abstraction supports flexible,
    robust subclassing for EKF, UKF, PSKF, OPSKF, etc.

    Attributes
    ----------
    sys : System
        The system model used for filtering. Exposes all validated model functions and metadata.

    Abstract Methods
    ----------------
    predict(xk, Pk, Fk, Qk, dt, uk, tk) → (x_pred, P_pred)
        Perform the prediction step of the Kalman filter at time `tk`.
        Must handle propagation of mean `xk` and covariance `Pk` using Jacobian `Fk`
        and process noise `Qk`.

    update(xk, Pk, yk, Hk, Rk, dt, uk, tk) → (x_upd, P_upd)
        Perform the measurement update using the observation `yk` at time `tk`.
        Must use the measurement Jacobian `Hk` and noise covariance `Rk`.

    run(x0, P0, Y, F, H, start_time, dt, U) → (X, P, T)
        Run the full filter over a measurement sequence `Y`.
        Applies prediction and update steps at each time step using linearizations `F`, `H`,
        and optionally time-varying inputs `U(t)`.
        Returns:
          - `X`: state estimates over time, shape (n_states, T)
          - `P`: state covariances over time, shape (n_states, n_states, T)
          - `T`: time vector, shape (T,)
    """

    def __init__(self, sys: System) -> None:
        self.sys = sys

    @abstractmethod
    def _predict(
        self,
        xk: NDArray,
        Pk: NDArray,
        Fk: NDArray,
        Qk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
    ) -> tuple[NDArray, NDArray]:
        """
        Perform the prediction step of the Kalman filter at time `tk`.
        """
        ...

    @abstractmethod
    def _update(
        self,
        xk: NDArray,
        Pk: NDArray,
        yk: NDArray,
        Hk: NDArray,
        Rk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
    ) -> tuple[NDArray, NDArray]:
        """
        Perform the measurement update using the observation `yk` at time `tk`.
        """
        ...

    @abstractmethod
    def run(
        self,
        *,
        x0: NDArray,
        P0: NDArray,
        Y: NDArray,
        F: Callable,
        H: Callable,
        start_time: float,
        dt: float,
        u: Optional[Callable],
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Run the full filter over a measurement sequence `Y`.
        """
        ...
