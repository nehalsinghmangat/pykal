import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Optional, Tuple
from pykal_core.utils.control_system.safeio import SafeIO

def kf_f(
    xk: Tuple[NDArray, NDArray],  # [x̂, P]
    tk: float,
    *,
    yk: NDArray,
    dt: float,
    f: Callable,
    F: Callable,
    Q: Callable,
    h: Callable,
    H: Callable,
    R: Callable,
    f_sys_type: str,
    kwargs_at_time_tk: Optional[Dict] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Extended Kalman Filter step (predict + update) using column-vector convention internally.

    Shapes:
        x̂_k : (n, 1)
        P_k  : (n, n)
        y_k  : (m, 1)
        F,H  : (n,n), (m,n)
        Q,R  : (n,n), (m,m)
    """

    if kwargs_at_time_tk is None:
        kwargs_at_time_tk = {}

    # --- helpers ------------------------------------------------------------
    def as_col(v: NDArray) -> NDArray:
        v = np.asarray(v)
        if v.ndim == 1:
            return v.reshape(-1, 1)
        if v.ndim == 2:
            if v.shape[1] == 1:
                return v
            if v.shape[0] == 1:
                return v.T
        raise ValueError(f"Expected vector-like array; got shape {v.shape}")

    def same_orientation(v_col: NDArray, like: NDArray) -> NDArray:
        like = np.asarray(like)
        if like.ndim == 1:
            return v_col.reshape(-1)
        if like.ndim == 2 and like.shape[0] == 1:
            return v_col.T
        return v_col

    # --- normalize inputs ---------------------------------------------------
    xhat_in, Pk = xk
    xhat = as_col(xhat_in)    # (n,1)
    yk_col = as_col(yk)       # (m,1)

    # --- PREDICT ------------------------------------------------------------
    Fk = SafeIO.smart_call(F, x=xhat, t=tk, kwargs_dict=kwargs_at_time_tk)
    Qk = SafeIO.smart_call(Q, x=xhat, t=tk, kwargs_dict=kwargs_at_time_tk)

    if f_sys_type in ("cti", "ctv"):
        xdot = SafeIO.smart_call(f, x=xhat, t=tk, kwargs_dict=kwargs_at_time_tk)
        xdot = as_col(xdot)
        x_pred = xhat + dt * xdot
        P_pred = Pk + dt * (Fk @ Pk + Pk @ Fk.T + Qk)

    elif f_sys_type in ("dti", "dtv"):
        x_pred = SafeIO.smart_call(f, x=xhat, t=tk, kwargs_dict=kwargs_at_time_tk)
        x_pred = as_col(x_pred)
        P_pred = Fk @ Pk @ Fk.T + Qk

    else:
        raise ValueError("f_sys_type must be one of ('cti','ctv','dti','dtv')")

    # --- UPDATE -------------------------------------------------------------
    y_pred = SafeIO.smart_call(h, x=x_pred, t=tk, kwargs_dict=kwargs_at_time_tk)
    y_pred = as_col(y_pred)

    Hk = SafeIO.smart_call(H, x=x_pred, t=tk, kwargs_dict=kwargs_at_time_tk)
    Rk = SafeIO.smart_call(R, x=x_pred, t=tk, kwargs_dict=kwargs_at_time_tk)

    innovation = yk_col - y_pred                       # (m,1)
    Sk = Hk @ P_pred @ Hk.T + Rk                       # (m,m)
    ridge = 1e-9 * np.eye(Sk.shape[0])
    try:
        Sk_inv = np.linalg.inv(Sk + ridge)
    except np.linalg.LinAlgError:
        Sk_inv = np.linalg.pinv(Sk + ridge)

    Kk = P_pred @ Hk.T @ Sk_inv                        # (n,m)
    x_upd = x_pred + Kk @ innovation                   # (n,1)

    I = np.eye(P_pred.shape[0])
    P_upd = (I - Kk @ Hk) @ P_pred @ (I - Kk @ Hk).T + Kk @ Rk @ Kk.T
    P_upd = 0.5 * (P_upd + P_upd.T)                    # symmetrize

    # --- return in original orientation -------------------------------------
    x_out = same_orientation(x_upd, xhat_in)
    return (x_out, P_upd)

def kf_sqrt_f():
    pass

def kf_h(xk: Tuple[NDArray, NDArray]) -> NDArray:
    # Identity measurement: returns current state estimate
    return xk[0]
