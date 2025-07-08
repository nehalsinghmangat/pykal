import scipy.linalg
import numpy as np
from typing import Callable, Optional, Sequence, Union
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from pykal.system import System


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

    All callable models are validated using `validate_system_function` from `utils.iosafety`
    and are accessed safely using `call_validated_function_with_args`.

    Attributes
    ----------
    sys : System
        The system model used for filtering. Exposes all validated model functions and metadata.
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

    ## Extenstion: Square-Root Factorization
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

    ## Extenstion: UD Factorization

    ## Extension: Partial Update
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
