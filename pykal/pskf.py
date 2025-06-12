import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from .utils.call import call
from .skf import SKF


class PSKF(SKF):
    """
    Partial-Update Schmidt-Kalman Filter (PSKF) [Brink 2017].

    Extends the Schmidt-Kalman filter by applying partial-update weights \u03B2 to each state.
    For simplicity, β must be provided as a pandas DataFrame of shape (N_steps, n_states),
    where N_steps is the number of filter update steps (i.e., length of measurement history).

    Parameters
    ----------
    f : Callable[..., np.ndarray]
        Process model function, f(x, u, t) -> x_dot or x_next.
    h : Callable[..., np.ndarray]
        Measurement model function, h(x, u, t) -> y.
    Q : np.ndarray, shape (n_states, n_states)
        Process noise covariance matrix.
    R : np.ndarray, shape (m_measurements, m_measurements)
        Measurement noise covariance matrix.
    beta : pd.DataFrame
        DataFrame of partial-update weights with shape (N_steps, n_states),
        indexed by the same time stamps used in `run` (i.e., start_time + np.arange(N_steps+1)*dt).
        Each entry in [0, 1].
    update_mask : Optional[np.ndarray], optional
        Boolean mask length n_states indicating which states to update; inherited from SKF.
    system_type : SystemType, optional
        Timing assumption for dynamics; default CONTINUOUS_TIME_INVARIANT.
    start_time : float, optional
        Starting time for the filter; default 0.0.
    dt : float, optional
        Time step for discrete systems; default 1.0.

    Attributes
    ----------
    beta_df : pd.DataFrame
        Time-varying beta DataFrame used for scaling gains.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pykal.pskf import PSKF
    >>> f = lambda x,u,t: np.zeros_like(x)
    >>> h = lambda x,u,t: x
    >>> Q = np.eye(2); R = 0.1 * np.eye(2)
    >>> times = [0.0, 1.0, 2.0]
    >>> beta_df = pd.DataFrame([[1.0, 0.5], [0.8, 0.2], [1.0, 1.0]], index=times)
    >>> pskf = PSKF(f=f, h=h, Q=Q, R=R, beta=beta_df)
    >>> x0 = pd.Series([0.0, 1.0], index=['a','b'])
    >>> P0 = np.eye(2)
    >>> y = pd.DataFrame([[0.1,1.1],[0.2,1.2],[0.3,1.3]], columns=['a','b'], index=times)
    >>> X_hist, _ = pskf.run(x0, P0, y, F=np.eye(2), H=np.eye(2))
    >>> list(X_hist.index)
    [0.0, 1.0, 2.0]
    >>> isinstance(pskf.beta_df, pd.DataFrame)
    True
    """
    def __init__(
        self,
        f,
        h,
        Q: np.ndarray,
        R: np.ndarray,
        beta: pd.DataFrame,
        update_mask: Optional[np.ndarray] = None,
        system_type=None,
        start_time: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        # Validate beta DataFrame
        if not isinstance(beta, pd.DataFrame):
            raise TypeError("beta must be a pandas DataFrame of shape (N_steps, n_states)")
        n_states = Q.shape[0]
        if beta.shape[1] != n_states:
            raise ValueError(f"beta DataFrame must have {n_states} columns, got {beta.shape[1]}")
        if beta.isnull().any().any():
            raise ValueError("beta DataFrame contains NaNs")
        if (beta.values < 0).any() or (beta.values > 1).any():
            raise ValueError("beta values must be in [0, 1]")

        # First call superclass (SKF) init
        super().__init__(
            f=f,
            h=h,
            Q=Q,
            R=R,
            system_type=system_type,
            start_time=start_time,
            dt=dt,
            update_mask=update_mask
        )
        # Store beta
        self.beta_df = beta.copy()

    def _get_beta(self, t: float) -> np.ndarray:
        """
        Retrieve the beta vector at time t from beta_df.

        Raises KeyError if t not in index.
        """
        try:
            row = self.beta_df.loc[t]
        except KeyError:
            raise ValueError(f"No beta row for time {t}")
        return row.values

    def update(
        self,
        x,
        P,
        y,
        H,
        t,
        u
    ):
        """
        Measurement update scaled by beta(t).

        Behaves like SKF.update but scales the gain rows by beta(t).
        """
        # Innovation covariance
        S = H @ P @ H.T + self.R
        ridge = 1e-9 * np.eye(S.shape[0])
        try:
            S_inv = np.linalg.inv(S + ridge)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S + ridge)

        # Compute Schmidt gain
        K_schmidt = super().compute_gain(P, H, S_inv)
        beta_vec = self._get_beta(t)
        K = beta_vec[:, None] * K_schmidt

        # Update
        z_pred = call(self.h, x=x, u=u, t=t)
        innov = y - z_pred
        x_upd = x + K @ innov
        I_KH = np.eye(P.shape[0]) - K @ H
        P_upd = I_KH @ P @ I_KH.T + K @ self.R @ K.T
        return x_upd, P_upd

    def plot_beta(self) -> plt.Axes:
        """
        Plot β over time from beta_df and return the Axes.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, col in enumerate(self.beta_df.columns):
            ax.plot(self.beta_df.index, self.beta_df[col], label=f"β[{col}]")
        ax.set_xlabel("Time")
        ax.set_ylabel("β")
        ax.set_title("PSKF Partial-Update Weights Over Time")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        return ax