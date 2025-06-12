import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Literal, Optional, Sequence, Tuple, Union
from .utils.call import call
from .utils.systemtype import SystemType
from .skf import SKF


class OPSKF(SKF):
    """
    Observability-informed Partial-Update Schmidt-Kalman Filter (OPSKF) [Ramos et al., 2021].

    This filter selects partial-update weights βi online based on an
    observability metric, either via the null-space of the linearized
    Gramian or a stochastic (covariance-eigenspace) proxy.

    Parameters
    ----------
    f : Callable[..., np.ndarray]
        Process model function, f(x, u, t) -> x_dot or x_next.
    h : Callable[..., np.ndarray]
        Measurement model function, h(x, u, t) -> y.
    Q : np.ndarray, shape (n_states, n_states)
        Process noise covariance.
    R : np.ndarray, shape (m_measurements, m_measurements)
        Measurement noise covariance.
    method : {'nullspace', 'stochastic'}, optional
        Strategy for β selection:
        - 'nullspace': projection onto least-observable null-space (Sec. III-A).
        - 'stochastic': projection onto principal covariance axis (Sec. III-B).
        Default: 'stochastic'.
    tol_null : float, optional
        Tolerance below which a direction is considered unobservable; used in nullspace method.
        Default: 1e-2.
    update_mask : Optional[np.ndarray], optional
        Boolean mask of length n_states indicating which states to correct (True) or hold constant.
        Inherited from SKF; default: all True.
    system_type : Optional[SystemType], optional
        Timing assumption for the system; default inherited.
    start_time : float, optional
        Filter start time; default 0.0.
    dt : float, optional
        Time step for discrete or discretized systems; default 1.0.

    Attributes
    ----------
    obs_metric_total : list[float]
        Total observability metric γ at each step.
    obs_metric_each : list[np.ndarray]
        Per-state normalized contributions |dx_i|/||dx|| at each step.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pykal.opskf import OPSKF
    >>> f = lambda x,u,t: x
    >>> h = lambda x,u,t: x
    >>> Q = np.eye(2); R = np.eye(2)
    >>> skf = OPSKF(f=f, h=h, Q=Q, R=R)
    >>> x0 = pd.Series([0.0,1.0], index=['a','b'])
    >>> P0 = np.eye(2)
    >>> y = pd.DataFrame([[0.1,1.1],[0.2,1.2]], columns=['a','b'])
    >>> X_hist, P_hist = skf.run(x0, P0, y, F=np.eye(2), H=np.eye(2))
    >>> hasattr(skf.obs_metric_total, 'append')
    True
    """
    def __init__(
        self,
        f: Callable[..., np.ndarray],
        h: Callable[..., np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        method: Literal['nullspace', 'stochastic'] = 'stochastic',
        tol_null: float = 1e-2,
        update_mask: Optional[np.ndarray] = None,
        system_type: SystemType = None,
        start_time: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__(f=f, h=h, Q=Q, R=R,
                         update_mask=update_mask,
                         system_type=system_type,
                         start_time=start_time, dt=dt)
        if method not in ('nullspace', 'stochastic'):
            raise ValueError("method must be 'nullspace' or 'stochastic'")
        self.method = method
        self.tol_null = float(tol_null)
        self.obs_metric_total: list[float] = []
        self.obs_metric_each: list[np.ndarray] = []

    def _beta_nullspace(
        self,
        P: np.ndarray,
        H: np.ndarray,
        K_full: np.ndarray,
        innov: np.ndarray
    ) -> np.ndarray:
        """
        Null-space β selection (Sec. III-A):
        projection onto the least-observable direction.

        Returns
        -------
        beta : np.ndarray, shape (n_states,)
        """
        G = H.T @ H
        eigvals, eigvecs = np.linalg.eigh(G)
        omin = eigvecs[:, np.argmin(eigvals)]
        dx = K_full @ innov
        proj = abs(float(dx @ omin)) / (np.linalg.norm(dx) + 1e-12)
        gamma = np.clip(proj, 0.0, 1.0)
        beta_val = 0.0 if gamma < self.tol_null else 1.0 - gamma
        return np.full(P.shape[0], beta_val)

    def _beta_stochastic(
        self,
        P: np.ndarray,
        K_full: np.ndarray,
        innov: np.ndarray
    ) -> np.ndarray:
        """
        Stochastic β selection (Sec. III-B):
        projection onto the principal covariance axis.

        Returns
        -------
        beta : np.ndarray, shape (n_states,)
        """
        eigvals, eigvecs = np.linalg.eigh(P)
        omin = eigvecs[:, np.argmax(eigvals)]
        dx = K_full @ innov
        proj = abs(float(dx @ omin)) / (np.linalg.norm(dx) + 1e-12)
        beta_val = np.clip(1.0 - proj, 0.0, 1.0)
        return np.full(P.shape[0], beta_val)

    def update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        t: Optional[float] = None,
        u: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform measurement update with dynamically chosen β.

        Records observability metrics and applies partial update.

        Returns
        -------
        x_upd : np.ndarray, shape (n_states,)
        P_upd : np.ndarray, shape (n_states, n_states)
        """
        # Innovation covariance
        S = H @ P @ H.T + self.R + 1e-9 * np.eye(H.shape[0])
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Full gain and innovation
        K_full = P @ H.T @ S_inv
        z_pred = call(self.h, x=x, u=u, t=t)
        innov = y - z_pred

        # Observability metrics
        dx = K_full @ innov
        if self.method == 'nullspace':
            G = H.T @ H
            eigvals, eigvecs = np.linalg.eigh(G)
            omin = eigvecs[:, np.argmin(eigvals)]
        else:
            eigvals, eigvecs = np.linalg.eigh(P)
            omin = eigvecs[:, np.argmax(eigvals)]
        proj_tot = abs(float(dx @ omin)) / (np.linalg.norm(dx) + 1e-12)
        obs_each = np.abs(dx) / (np.linalg.norm(dx) + 1e-12)
        self.obs_metric_total.append(proj_tot)
        self.obs_metric_each.append(obs_each)

        # Dynamic β and partial gain
        if self.method == 'nullspace':
            beta = self._beta_nullspace(P, H, K_full, innov)
        else:
            beta = self._beta_stochastic(P, K_full, innov)
        D = np.diag(beta)
        K_part = D @ K_full

        # Update state and covariance
        x_upd = x + K_part @ innov
        I_KH = np.eye(len(x)) - K_part @ H
        P_upd = I_KH @ P @ I_KH.T + K_part @ self.R @ K_part.T

        # Freeze nuisance states
        mask = ~self.update_mask
        x_upd[mask] = x[mask]
        P_upd[mask, :] = P[mask, :]
        P_upd[:, mask] = P[:, mask]

        return x_upd, P_upd

    def plot_observability(
        self,
        mode: Literal['total', 'individual'] = 'total'
    ) -> Union[plt.Axes, Sequence[plt.Axes]]:
        """
        Plot the recorded observability metrics and return the Axes.

        Parameters
        ----------
        mode : {'total', 'individual'}, optional
            'total'      – plot total metric γ over time
            'individual' – plot per-state normalized contributions over time
            Default: 'total'.

        Returns
        -------
        ax : matplotlib.axes.Axes or sequence of Axes
        """
        if not self.obs_metric_total:
            raise RuntimeError("No observability data: run the filter first.")
        t = np.arange(len(self.obs_metric_total))

        if mode == 'total':
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t, self.obs_metric_total, marker='o', label='Total γ')
            ax.axhline(self.tol_null, linestyle='--', label=f'tol_null={self.tol_null}')
            ax.set_xlabel('Time step')
            ax.set_ylabel('γ')
            ax.set_title('OPSKF Total Observability Metric')
            ax.legend()
            ax.grid(True)
            return ax

        # individual mode
        obs_arr = np.vstack(self.obs_metric_each)
        n_states = obs_arr.shape[1]
        fig, axes = plt.subplots(n_states, 1, figsize=(8, 3 * n_states), sharex=True)
        if n_states == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(t, obs_arr[:, i], label=f'State {i}')
            ax.set_ylabel(f'|dx[{i}]|/||dx||')
            ax.legend()
            ax.grid(True)
        axes[-1].set_xlabel('Time step')
        return axes
