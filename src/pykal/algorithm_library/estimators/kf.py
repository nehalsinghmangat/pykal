import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Callable, Dict, Tuple
from pykal.dynamical_system import DynamicalSystem


class KF:
    @staticmethod
    def f(
        *,
        zk: Tuple[NDArray, NDArray],
        yk: NDArray,
        f: Callable,
        f_h_params: Dict,
        h: Callable,
        Fk: NDArray,
        Qk: NDArray,
        Hk: NDArray,
        Rk: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Perform one full **predict–update** step of the discrete-time Kalman Filter.
        """

        # === Extract covariance ===
        _, Pk = zk

        # === Predict ===
        x_pred = DynamicalSystem._smart_call(f, f_h_params)
        P_pred = Fk @ Pk @ Fk.T + Qk

        # === Innovation ===
        y_pred = DynamicalSystem._smart_call(h, f_h_params)
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
    def h(zk: Tuple[NDArray, NDArray]) -> NDArray:
        # extracts current state estimate
        return zk[0]


# Module-level aliases for convenience
# Allows usage like: from pykal.algorithm_library.estimators import kf; kf.f(...)
f = KF.f
h = KF.h
