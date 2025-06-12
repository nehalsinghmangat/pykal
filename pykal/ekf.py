import numpy as np
from typing import Callable, Optional, Tuple
from .utils.call import call
from .utils.systemtype import SystemType
from .base import BaseKF


class EKF(BaseKF):
    """
    Extended Kalman Filter (EKF).

    This filter extends the linear Kalman filter to nonlinear systems by
    linearizing the process and measurement models around the current estimate.

    Parameters
    ----------
    f : Callable[..., np.ndarray]
        Process model function, f(x, u, t) -> x_dot or x_next.
    h : Callable[..., np.ndarray]
        Measurement model function, h(x, u, t) -> y.
    Q : np.ndarray
        Process noise covariance matrix of shape (n_states, n_states).
    R : np.ndarray
        Measurement noise covariance matrix of shape (m_measurements, m_measurements).
    system_type : SystemType, optional
        Timing assumption for the system (continuous vs discrete); default CONTINUOUS_TIME_INVARIANT.
    start_time : float, optional
        Starting time for the filter; default 0.0.
    dt : float, optional
        Time step for discrete or discretized systems; default 1.0.

    Examples
    --------            
    >>> import numpy as np, pandas as pd    
    >>> from pykal.ekf import EKF    
    >>> f = lambda x, u, t: x
    >>> h = lambda x, u, t: x
    >>> Q = np.eye(2); R = np.eye(2)
    >>> ekf = EKF(f=f, h=h, Q=Q, R=R)
    >>> x0 = pd.Series([0.0, 1.0], index=['a', 'b'])
    >>> P0 = np.eye(2)
    >>> y = pd.DataFrame([[0.1, 1.1], [0.2, 1.2]], columns=['a','b'])
    >>> X_hist, P_hist = ekf.run(x0, P0, y, F=np.eye(2), H=np.eye(2))
    >>> X_hist.shape
    (3, 2)
    """
    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        system_type: SystemType = SystemType.CONTINUOUS_TIME_INVARIANT,
        start_time: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__(f=f, h=h, Q=Q, R=R, system_type=system_type, start_time=start_time, dt=dt)

    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        F: np.ndarray,
        t: Optional[float],
        u: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time update (prediction) step of the EKF.

        For continuous-time dynamics:
        x_pred = x + f(x,u,t) * dt
        P_pred = P + (F @ P + P @ F^T + Q) * dt

        For discrete-time dynamics:
        x_pred = f(x,u,t)
        P_pred = F @ P @ F^T + Q

        Parameters
        ----------
        x : np.ndarray, shape (n_states,)
            Current state estimate.
        P : np.ndarray, shape (n_states, n_states)
            Current estimate covariance.
        F : np.ndarray, shape (n_states, n_states)
            Jacobian of the process model w.r.t. state.
        t : float or None
            Time at which to evaluate f and F.
        u : np.ndarray or None
            Control input at time t.

        Returns
        -------
        x_pred : np.ndarray, shape (n_states,)
            Predicted state.
        P_pred : np.ndarray, shape (n_states, n_states)
            Predicted covariance.

        Examples
        --------            
        >>> import numpy as np        
        >>> from pykal.ekf import EKF    
        >>> ekf = EKF(f=lambda x,u,t: np.ones_like(x), h=lambda **kw: x, Q=np.eye(1), R=np.eye(1), system_type=SystemType.DISCRETE_TIME_INVARIANT)
        >>> x_pred, P_pred = ekf.predict(np.array([1.0]), np.eye(1), np.eye(1), t=0.0, u=None)
        >>> x_pred.shape, P_pred.shape
        ((1,), (1, 1))
        """
        if self.system_type in (
            SystemType.CONTINUOUS_TIME_INVARIANT,
            SystemType.CONTINUOUS_TIME_VARYING
        ):
            x_pred = x + call(self.f, x=x, u=u, t=t) * self.dt
            P_pred = P + (F @ P + P @ F.T + self.Q) * self.dt
        else:
            # DISCRETE_TIME_INVARIANT or DISCRETE_TIME_VARYING
            x_pred = call(self.f, x=x, u=u, t=t)
            P_pred = F @ P @ F.T + self.Q

        return x_pred, P_pred

    def compute_gain(
        self,
        P: np.ndarray,
        H: np.ndarray,
        S_inv: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Kalman gain.

        K = P @ H^T @ S_inv

        Parameters
        ----------
        P : np.ndarray, shape (n_states, n_states)
            Predicted covariance.
        H : np.ndarray, shape (m_measurements, n_states)
            Jacobian of the measurement model.
        S_inv : np.ndarray, shape (m_measurements, m_measurements)
            Inverse of the innovation covariance.

        Returns
        -------
        K : np.ndarray, shape (n_states, m_measurements)
            Kalman gain matrix.

        Examples
        --------            
        >>> import numpy as np        
        >>> from pykal.ekf import EKF    
        >>> ekf = EKF(f=lambda **kw: None, h=lambda **kw: None, Q=np.eye(1), R=np.eye(1))
        >>> K = ekf.compute_gain(np.eye(1), np.eye(1), np.eye(1))
        >>> float(K[0,0])
        1.0
        """
        return P @ H.T @ S_inv

    def update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        t: Optional[float],
        u: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measurement update step of the EKF.

        Computes innovation covariance, Kalman gain, and applies the update:
        S = H @ P @ H^T + R
        K = P @ H^T @ S^{-1}
        x_upd = x + K @ (y - h(x,u,t))
        P_upd = (I - K H) P (I - K H)^T + K R K^T

        Parameters
        ----------
        x : np.ndarray, shape (n_states,)
            Predicted state.
        P : np.ndarray, shape (n_states, n_states)
            Predicted covariance.
        y : np.ndarray, shape (m_measurements,)
            Measurement vector.
        H : np.ndarray, shape (m_measurements, n_states)
            Jacobian of the measurement model.
        t : float or None
            Time at which to evaluate h and H.
        u : np.ndarray or None
            Control input at time t.

        Returns
        -------
        x_upd : np.ndarray, shape (n_states,)
            Updated state estimate.
        P_upd : np.ndarray, shape (n_states, n_states)
            Updated covariance estimate.

        Examples
        --------
        >>> import numpy as np        
        >>> from pykal.ekf import EKF        
        >>> ekf = EKF(f=lambda **kw: None, h=lambda x,u,t: x, Q=np.eye(1), R=np.eye(1))
        >>> x_upd, P_upd = ekf.update(np.array([1.0]), np.eye(1), np.array([2.0]), np.eye(1), t=0.0, u=None)
        >>> x_upd.shape, P_upd.shape
        ((1,), (1, 1))            
        """
        S = H @ P @ H.T + self.R
        ridge = 1e-9 * np.eye(S.shape[0])
        try:
            S_inv = np.linalg.inv(S + ridge)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S + ridge)

        K = self.compute_gain(P, H, S_inv)
        z_pred = call(self.h, x=x, u=u, t=t)
        x_upd = x + K @ (y - z_pred)
        I_KH = np.eye(P.shape[0]) - K @ H
        P_upd = I_KH @ P @ I_KH.T + K @ self.R @ K.T

        return x_upd, P_upd