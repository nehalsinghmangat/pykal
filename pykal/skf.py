import numpy as np
from typing import Optional
from .utils.systemtype import SystemType
from .ekf import EKF


class SKF(EKF):
    """
    Schmidt-Kalman Filter (SKF).

    The SKF applies Schmidt's partial update: only a subset of ``active`` states
    (specified by ``update_mask``) are corrected by each measurement; nuisance
    states remain frozen at their prior estimates.

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
        Timing assumption for dynamics; default CONTINUOUS_TIME_INVARIANT.
    start_time : float, optional
        Starting time for the filter; default 0.0.
    dt : float, optional
        Time step for discrete or discretized systems; default 1.0.
    update_mask : Optional[np.ndarray], optional
        Boolean mask of length n_states indicating which states to update
        (``True`` = update, ``False`` = hold constant). Default: all True.

    Attributes
    ----------
    update_mask : np.ndarray
        The boolean mask applied to the Kalman gain to freeze nuisance states.

    Examples
    --------
    >>> import numpy as np
    >>> from pykal.skf import SKF
    >>> # Define trivial system where f and h are identity
    >>> f = lambda x, u, t: x
    >>> h = lambda x, u, t: x
    >>> Q = np.eye(2); R = np.eye(2)
    >>> # Mask: update only first state
    >>> mask = np.array([True, False])
    >>> skf = SKF(f=f, h=h, Q=Q, R=R, update_mask=mask)
    >>> x = np.array([1.0, 2.0])
    >>> P = np.eye(2)
    >>> y = np.array([1.1, 2.1])
    >>> H = np.eye(2)
    >>> # Compute gain with identity innovation covariance
    >>> S_inv = np.eye(2)
    >>> K = skf.compute_gain(P, H, S_inv)
    >>> K.shape
    (2, 2)
    >>> # Second row should be zero due to mask
    >>> np.all(K[1] == 0)
    True
    """        
    def __init__(
        self,
        f,
        h,
        Q: np.ndarray,
        R: np.ndarray,
        system_type: SystemType = SystemType.CONTINUOUS_TIME_INVARIANT,
        start_time: float = 0.0,
        dt: float = 1.0,
        update_mask: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(f=f, h=h, Q=Q, R=R,
                         system_type=system_type,
                         start_time=start_time, dt=dt)
        n_states = self.Q.shape[0]
        if update_mask is None:
            update_mask = np.ones(n_states, dtype=bool)
        if update_mask.shape != (n_states,):
            raise ValueError(
                f"update_mask must be 1-D of length {n_states}; got {update_mask.shape}"
            )
        self.update_mask = update_mask

    def compute_gain(
        self,
        P: np.ndarray,
        H: np.ndarray,
        S_inv: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Schmidt-Kalman gain.

        This is the standard EKF gain:

            K = P @ H.T @ S_inv

        with nuisance rows (``update_mask == False``) zeroed to freeze those states.

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
            Schmidt-Kalman gain with nuisance-state rows zeroed.

        Examples
        --------
        >>> import numpy as np
        >>> from pykal.skf import SKF
        >>> f = lambda x,u,t: x; h = lambda x,u,t: x
        >>> Q = np.eye(1); R = np.eye(1)
        >>> skf = SKF(f, h, Q, R, update_mask=[False])
        >>> P = np.eye(1); H = np.eye(1); S_inv = np.eye(1)
        >>> K = skf.compute_gain(P, H, S_inv)
        >>> K[0,0]
        0.0            
        """
        K = super().compute_gain(P, H, S_inv)
        K[~self.update_mask, :] = 0.0
        return K
