from functools import wraps
from typing import Callable, Optional, Sequence, Union
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from system import System
import scipy.linalg
import numpy as np
from system import System
from utils.utils_safeio import SafeIO as safeio


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
        square_root: bool,
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
        y_pred: NDArray,
        Hk: NDArray,
        Rk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
        square_root: bool,
        beta_mat: NDArray,
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
        square_root: bool,
        override_beta: Optional[Callable],
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Run the full filter over a measurement sequence `Y`.
        """
        ...


class BaseKFSqrt(BaseKF):
    def __init__(self, sys: System, **kwargs):
        super().__init__(sys)

    def _square_root_predict_covariance(
        self,
        Pk: NDArray,
        Fk: NDArray,
        Qk: NDArray,
    ) -> NDArray:
        """
        Square-root prediction via QR factorization.
        """
        try:
            ridge = 1e-9 * np.eye(Pk.shape[0])
            S = scipy.linalg.cholesky(Pk + ridge, lower=True)
            Q_sqrt = scipy.linalg.cholesky(Qk + ridge, lower=True)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "Cholesky decomposition failed in square-root predict."
            ) from e

        FS = Fk @ S  # shape (n, n)

        # Pad Q_sqrt horizontally if necessary to match FS
        if Q_sqrt.shape[1] < FS.shape[1]:
            pad = FS.shape[1] - Q_sqrt.shape[1]
            Q_sqrt = np.hstack([Q_sqrt, np.zeros((Q_sqrt.shape[0], pad))])
        elif Q_sqrt.shape[1] > FS.shape[1]:
            raise ValueError(
                f"Q_sqrt has too many columns: {Q_sqrt.shape[1]} > {FS.shape[1]}"
            )

        A = np.hstack((FS, Q_sqrt))  # shape (n, 2n)
        _, R = np.linalg.qr(A.T)
        P_upd = R.T @ R
        return 0.5 * (P_upd + P_upd.T)  # symmetrize

    def _square_root_update_covariance(
        self,
        Pk: NDArray,
        Hk: NDArray,
        Rk: NDArray,
    ) -> NDArray:
        """
        Square-root update via QR factorization.
        """
        try:
            ridge = 1e-9 * np.eye(Pk.shape[0])
            S = scipy.linalg.cholesky(Pk + ridge, lower=True)
            R_sqrt = scipy.linalg.cholesky(Rk + ridge, lower=True)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "Cholesky decomposition failed in square-root update."
            ) from e

        HS = Hk @ S  # shape (m, n)

        # Pad R_sqrt horizontally to match HS width
        if R_sqrt.shape[1] < HS.shape[1]:
            pad = HS.shape[1] - R_sqrt.shape[1]
            R_sqrt = np.hstack([R_sqrt, np.zeros((R_sqrt.shape[0], pad))])
        elif R_sqrt.shape[1] > HS.shape[1]:
            raise ValueError(
                f"R_sqrt has too many columns: {R_sqrt.shape[1]} > {HS.shape[1]}"
            )

        A = np.vstack((HS, R_sqrt))  # shape (2m, n)
        _, R = np.linalg.qr(A)
        R = R[: Pk.shape[0], : Pk.shape[0]]  # ensure (n, n) shape
        P_upd = R.T @ R
        return 0.5 * (P_upd + P_upd.T)  # symmetrize


class BaseKFPartialUpdate(BaseKF):
    """
    Abstract base class for Kalman Filters supporting per-state partial updates.

    This class adds a validated β function to control the update strength of each
    state during the correction step. Each βₖ ∈ [0, 1]^n is a column vector that is
    broadcast into a diagonal matrix used to weight the update.

    Subclasses should invoke `self._beta(t)` to retrieve βₖ at each time step.

    Attributes
    ----------
    _beta : Callable[[float], NDArray]
        A function β(t) → NDArray of shape (n, 1), where each entry ∈ [0, 1].
        This is converted to a diagonal weighting matrix during the update.
    """

    def __init__(self, sys: System, beta: Optional[Callable] = None, **kwargs):
        super().__init__(sys)
        self._beta = self.set_beta(beta, self.sys.state_names)

    @property
    def beta(self) -> Callable:
        """Return the current β function."""
        return self._beta

    @beta.setter
    def beta(self, func: Callable):
        """Set a new β function and validate its output shape."""
        self._beta = self.set_beta(func, self.sys.state_names)

    def set_beta(
        self, beta: Union[Callable, None, bool], state_names: Sequence[str]
    ) -> Callable:
        """
        Set the β function for the Partial-Update Kalman Filter (PKF).

        This function defines how much each individual state is updated during the EKF
        correction step. The user may provide a callable β(t) → NDArray of shape (n, 1),
        where `n = len(state_names)`. If `beta` is None, a default function is returned
        that always outputs a column vector of ones (full update).

        Parameters
        ----------
        beta : Callable or None
            A function of the form `beta(t) -> NDArray` of shape (n, 1), or `None` to
            use the default (full update) behavior.
        state_names : Sequence[str]
            List of state names used to determine the output shape of the default β.

        Returns
        -------
        Callable
            A validated β function returning an array of shape (n, 1), where n = number of states.
        """
        if beta is None:
            n_states = len(state_names)

            def default_beta(t: float) -> NDArray:
                return np.ones((n_states, 1))

            return default_beta
        else:

            @safeio.verify_signature_and_parameter_names
            def verify_beta(beta: Callable) -> Callable:
                return beta

            return verify_beta(beta)

    def _apply_partial_update(
        self, P_base: NDArray, Pk: NDArray, tk: float, beta_mat: NDArray
    ) -> NDArray:
        """
        Apply per-state partial update weighting to the covariance matrix.

        This implements the correction:
            P_upd = β P_base βᵀ + (I - β) Pk (I - β)ᵀ
        where β ∈ ℝⁿˣⁿ is a diagonal matrix of per-state update strengths in [0, 1].

        Parameters
        ----------
        P_base : NDArray
            Base covariance from EKF/UKF update, shape (n, n)
        Pk : NDArray
            Prior covariance before update, shape (n, n)
        tk : float
            Time at which to evaluate β(t)

        Returns
        -------
        P_upd : NDArray
            Weighted covariance update, shape (n, n)
        """
        I = np.eye(beta_mat.shape[0])
        return beta_mat @ P_base @ beta_mat.T + (I - beta_mat) @ Pk @ (I - beta_mat).T
