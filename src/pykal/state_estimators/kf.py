import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple


class KF:
    @staticmethod
    def standard_f(
        *,
        xhat_P: Tuple[NDArray, NDArray],
        yk: NDArray,
        f: Callable,
        f_params: Dict,
        h: Callable,
        h_params: Dict,
        Fk: NDArray,
        Qk: NDArray,
        Hk: NDArray,
        Rk: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Perform one full **predict–update** step of the discrete-time Kalman Filter.

        Parameters
        ----------
        xhat_P : Tuple[NDArray, NDArray]
            A tuple ``(x_hat_k, P_k)`` containing:
                - ``x_hat_k`` : the current estimated state of the plant, shape (n,1)
                - ``P_k``     : the current state covariance, shape (n,n)

        yk : NDArray
            The measurement at time k, shape (m,1).

        f : Callable
            The plant evolution function used to propagate the estimated mean:
                ``x_pred = f(**f_params)``
            This should return the **noise-free** predicted state.

        f_params : Dict
            Dictionary of parameters passed to the evolution function ``f``.

        h : Callable
            The plant output function used to compute the predicted measurement:
                ``y_pred = h(**h_params)``
            This should return the **noise-free** predicted measurement.

        h_params : Dict
            Dictionary of parameters passed to the measurement function ``h``.

        Fk : NDArray
            The state-transition Jacobian evaluated at the current estimate,
            used to propagate the covariance. Shape (n,n).

        Qk : NDArray
            The process-noise covariance matrix at time k, shape (n,n).

        Hk : NDArray
            The measurement Jacobian evaluated at the current estimate,
            used in the update. Shape (m,n).

        Rk : NDArray
            The measurement-noise covariance matrix at time k, shape (m,m).


        Returns
        -------
        (x_upd, P_upd) : Tuple[NDArray, NDArray]
            The updated state estimate and covariance:
                - ``x_upd`` : updated state estimate, shape (n,1)
                - ``P_upd`` : updated state covariance, shape (n,n)


        Notes
        -----
        This implementation follows the standard **linearized EKF equations**:

        **Predict step**
        ----------------
        State prediction:
            ``x_pred = f(**f_params)``

        Covariance prediction:
            ``P_pred = Fk @ Pk @ Fk.T + Qk``

        **Update step**
        ---------------
        Innovation:
            ``innovation = yk - y_pred``
            where ``y_pred = h(**h_params)``

        Innovation covariance:
            ``Sk = Hk @ P_pred @ Hk.T + Rk``

        Kalman gain:
            ``Kk = P_pred @ Hk.T @ Sk^{-1}``

        State update:
            ``x_upd = x_pred + Kk @ innovation``

        Covariance update (Joseph form for numerical stability):
            ``P_upd = (I - Kk @ Hk) @ P_pred @ (I - Kk @ Hk).T + Kk @ Rk @ Kk.T``

        The covariance matrix is explicitly symmetrized at the end to counter
        numerical drift:
            ``P_upd = 0.5 * (P_upd + P_upd.T)``
        """

        # === Extract covariance ===
        _, Pk = xhat_P

        # === Predict ===
        x_pred = f(**f_params)
        P_pred = Fk @ Pk @ Fk.T + Qk

        # === Innovation ===
        y_pred = h(**h_params)
        innovation = yk - y_pred

        # === Update ===
        Sk = Hk @ P_pred @ Hk.T + Rk
        ridge = 1e-9 * np.eye(Sk.shape[0])
        try:
            Sk_inv = np.linalg.inv(Sk + ridge)
        except np.linalg.LinAlgError:
            Sk_inv = np.linalg.pinv(Sk + ridge)

        Kk = P_pred @ Hk.T @ Sk_inv
        x_upd = x_pred + Kk @ innovation

        I = np.eye(P_pred.shape[0])
        P_upd = (I - Kk @ Hk) @ P_pred @ (I - Kk @ Hk).T + Kk @ Rk @ Kk.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        return (x_upd, P_upd)

    @staticmethod
    def standard_h(xhat_P: Tuple[NDArray, NDArray]) -> NDArray:
        # extracts current state estimate
        return xhat_P[0]
