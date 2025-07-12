import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional


def pass_partial_update_beta_into_partial_update_override_function(
    beta: Callable,
) -> Callable:

    def partial_update_override(
        *,
        xk: NDArray,
        uk: NDArray,
        tk: float,
        Pk: NDArray,
        Hk: NDArray,
        Rk: NDArray,
        Kk: NDArray,
        innovation: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Partial-update Kalman filter override.

        Parameters
        ----------
        x : (n, 1) np.ndarray
            Prior state estimate.
        P : (n, n) np.ndarray
            Prior covariance matrix.
        y : (m, 1) np.ndarray
            Current measurement.
        H : (m, n) np.ndarray
            Measurement Jacobian.
        R : (m, m) np.ndarray
            Measurement noise covariance.
        K : (n, m) np.ndarray
            Kalman gain.
        innovation : (m, 1) np.ndarray
            Measurement residual.
        beta : (n, n) np.ndarray
            Partial update matrix.

        Returns
        -------
        x_upd : (n, 1) np.ndarray
            Updated state estimate.
        P_upd : (n, n) np.ndarray
            Updated covariance matrix.
        """

        beta_matrix = np.diagflat((beta(Pk=Pk)))
        I = np.eye(Pk.shape[0])
        x_upd = xk + beta_matrix @ Kk @ innovation
        P_upd = (I - beta_matrix @ Kk @ Hk) @ Pk @ (
            I - beta_matrix @ Kk @ Hk
        ).T + beta_matrix @ Kk @ Rk @ Kk.T @ beta_matrix.T
        return x_upd, P_upd

    return partial_update_override
