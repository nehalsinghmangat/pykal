import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from matplotlib import pyplot as plt
from typing import Callable, Optional, Sequence,Union
from abc import ABC, abstractmethod
from .utils.call import call
from .utils.systemtype import SystemType



class BaseKF(ABC):
    """
    Abstract base class for Kalman-filter estimators.

    This class defines the common interface and `run` loop for all Kalman Filter
    variants (standard, Extended, Schmidt, PSKF, OPSKF).  Subclasses must implement
    the `predict` and `update` methods.

    Parameters
    ----------
    f, h : Callable
        Process (`f`) and measurement (`h`) model functions. Each may accept any
        subset of the keyword args `x`, `u`, `t`, or their aliases — see
        :func:`pykal.utils.call.call` for the full list of supported parameter names.
    Q : np.ndarray
        Process noise covariance matrix of shape ``(n_states, n_states)``.
    R : np.ndarray
        Measurement noise covariance matrix of shape
        ``(m_measurements, m_measurements)``.
    system_type : :class:`pykal.utils.systemtype.SystemType`, default=SystemType.CONTINUOUS_TIME_INVARIANT
        Timing assumption for dynamics (continuous vs. discrete time); see
        the `SystemType` enum for options.
    dt : float, default=1.0
        Time step (seconds) for discrete-time models.

    Attributes
    ----------
    f, h, Q, R, system_type, dt
        Stored copies of the constructor arguments.

    Methods
    -------
    predict(x, P, F, t, u)
        Abstract: perform the time update.
    update(x, P, y, H, t, u)
        Abstract: perform the measurement update.
    run(x0, P0, y, F, H, u=None)
        Iterate predict+update over a sequence of measurements.
    plot_kf_predictions(...)
        Plot state estimates, true states, measurements, and uncertainty.
    plot_error(...)
        Plot RMSE and NEES consistency metrics.
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
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.system_type = system_type
        self.start_time = float(start_time)
        self.dt = float(dt)

    @abstractmethod
    def predict(self, 
                x: np.ndarray, 
                P: np.ndarray, 
                F: Callable, 
                t: Optional[float],
                u: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def update(self, 
               x: np.ndarray, 
               P: np.ndarray, 
               y: np.ndarray,
               H: Callable,
               t: Optional[float],
               u: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        ...
    def run(
        self,
        x0: pd.Series,
        P0: np.ndarray,
        y: pd.DataFrame,
        F: Union[np.ndarray, Callable],
        H: Union[np.ndarray, Callable],
        u: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Iterate the filter over a sequence of measurements and return labeled histories.

        Parameters
        ----------
        x0 : pd.Series, shape (n_states,)
        Initial state vector; index corresponds to state names.
        P0 : np.ndarray, shape (n_states, n_states)
        Initial state covariance.
        y : pd.DataFrame, shape (N_steps, m_measurements)
        Measurement history; index is ignored (time is inferred).
        F : Union[np.ndarray, Callable]
        State transition matrix `(n_states, n_states)` or callable
        `F(x, u=None, t=None) → np.ndarray` of that shape.
        H : Union[np.ndarray, Callable]
        Measurement matrix `(m_measurements, n_states)` or callable
        `H(x, u=None, t=None) → np.ndarray` of that shape.
        u : pd.DataFrame, optional
        Control-input history; if shorter than `y`, last row is held constant.

        Returns
        -------
        X_hist : pd.DataFrame, shape (N_steps+1, n_states)
        State estimates over time, indexed by absolute time stamps.
        P_hist : pd.DataFrame, shape (N_steps+1, n_states*n_states)
        Covariance history flattened along columns `(state_i, state_j)`,
        indexed by absolute time stamps.

        Raises
        ------
        ValueError
        If input dimensions are invalid or if `y` is empty.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from pykal.base import BaseKF
        >>> from pykal.utils.systemtype import SystemType
        >>> class IdentityKF(BaseKF):
        ...     def predict(self, x, P, F, t, u): return x, P
        ...     def update(self, x, P, y, H, t, u): return x, P
        >>> x0 = pd.Series([0.0, 1.0], index=['a', 'b'])
        >>> P0 = np.eye(2)
        >>> y = pd.DataFrame([[1.1, 2.2], [3.3, 4.4]], columns=['a', 'b'])
        >>> kf = IdentityKF(
        ...     f=lambda **kw: None,
        ...     h=lambda **kw: None,
        ...     Q=np.eye(2), R=np.eye(2),
        ...     system_type=SystemType.DISCRETE_TIME_INVARIANT,
        ...     start_time=0.0, dt=1.0
        ... )
        >>> X_hist, P_hist = kf.run(x0, P0, y, F=np.eye(2), H=np.eye(2))
        >>> isinstance(X_hist, pd.DataFrame)
        True
        >>> X_hist.shape
        (3, 2)
        >>> isinstance(P_hist, pd.DataFrame)
        True
        >>> P_hist.shape
        (3, 4)
        >>> list(P_hist.columns)[:2] == [('a','a'), ('a','b')]
        True
        """
        # --- input validation ---
        if not isinstance(x0, pd.Series):
            raise ValueError(f"x0 is type {type(x0)!r}; must be pd.Series")
        if not isinstance(P0, np.ndarray):
            raise ValueError(f"P0 is type {type(P0)!r}; must be np.ndarray")
        if not isinstance(y, pd.DataFrame):
            raise ValueError(f"y is type {type(y)!r}; must be pd.DataFrame")

        n_states = x0.shape[0]
        N_steps  = y.shape[0]
        m_meas    = y.shape[1]

        if P0.shape != (n_states, n_states):
            raise ValueError(f"P0 must be ({n_states},{n_states}); got {P0.shape}")
        if N_steps == 0:
            raise ValueError("Measurement history `y` must contain at least one row")
        if m_meas == 0:
            raise ValueError("Measurement history `y` must contain at least one column")

        # --- pre-allocate histories ---
        x_hist_arr = np.empty((N_steps + 1, n_states))
        P_hist_arr = np.empty((N_steps + 1, n_states, n_states))

        # --- initialize arrays ---
        x_hist_arr[0] = x0.values.copy()
        P_hist_arr[0] = P0.copy()

        # --- precompute time axis ---
        t_hist_arr = self.start_time + np.arange(N_steps + 1) * self.dt

        # --- main loop ---
        xk = x_hist_arr[0]
        Pk = P_hist_arr[0]
        y_vals = y.to_numpy()
        for k in range(N_steps):
            yk  = y_vals[k]
            t_k = t_hist_arr[k]         

            # zero-order-hold for inputs
            if u is None or len(u) == 0:
                uk = None
            else:
                uk = u.iloc[min(k, len(u) - 1)].values
                if k >= len(u):
                    warnings.warn(
                        "Control shorter than measurements; holding last input."
                    )

            Fk = call(F, x=xk, u=uk, t=t_k) if callable(F) else F
            Hk = call(H, x=xk, u=uk, t=t_k) if callable(H) else H

            if Fk.shape != (n_states, n_states):
                raise ValueError(f"F must be ({n_states},{n_states}), got {Fk.shape} at time {t_k}")
            if Hk.shape != (m_meas, n_states):
                raise ValueError(f"H must be ({m_meas},{n_states}), got {Hk.shape} at time {t_k}")

            xk, Pk = self.predict(xk, Pk, Fk, t_k, uk)
            xk, Pk = self.update(xk, Pk, yk, Hk, t_k, uk)

            x_hist_arr[k + 1] = xk
            P_hist_arr[k + 1] = Pk

        # --- build DataFrames with time index ---
        states   = list(x0.index)
        time_idx = pd.Index(t_hist_arr, name="time")

        X_hist = pd.DataFrame(x_hist_arr, index=time_idx, columns=states)

        mi     = pd.MultiIndex.from_product([states, states],
                                            names=["state_i", "state_j"])
        P_flat = P_hist_arr.reshape((N_steps + 1, -1))
        P_hist = pd.DataFrame(P_flat, index=time_idx, columns=mi)

        return X_hist, P_hist

    def plot_kf_predictions(
    self,
    x_hist: pd.DataFrame,
    P_hist: pd.DataFrame,
    x: Optional[pd.DataFrame] = None,
    y: Optional[pd.DataFrame] = None,
    inputs_df: Optional[pd.DataFrame] = None,
    include_sigma_uncertainty: bool = True,
    n_sigma_bound: float = 3.0,
    ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None
    ) -> Union[plt.Axes, Sequence[plt.Axes]]:
        """
        Plot Kalman filter state estimate history (and optional truth/measurements/inputs)
        on provided axes or create new ones, and return the axes for further customization.

        Parameters
        ----------
        x_hist : pd.DataFrame, shape (T+1, n_states)
            Filtered state estimates over time; index should be time stamps.
        P_hist : pd.DataFrame, shape (T+1, n_states*n_states)
            Filtered covariance history as a MultiIndex DataFrame with
            column levels ('state_i','state_j').
        x : pd.DataFrame, optional
            True state history; indexed by the same time stamps as `x_hist`. Only columns
            matching state names in `x_hist` will be plotted; any extra columns are ignored.
        y : pd.DataFrame, optional
            Measurement history; indexed by the same time stamps as `x_hist`. Columns
            corresponding to states in `x_hist` will be plotted as measurements. Measurements
            do not have to cover all states—only those present will be shown.
        inputs_df : pd.DataFrame, optional
            Control-input history; indexed like `x_hist`. Only matching columns are plotted.
        include_sigma_uncertainty : bool, default=True
            Whether to draw ±n_sigma_bound uncertainty bands.
        n_sigma_bound : float, default=3.0
            Number of standard deviations for uncertainty bands.
        ax : Axes or sequence of Axes, optional
            Axes to plot on. If None, new subplots are created; if provided,
            length must equal number of states.

        Returns
        -------
        axs : Axes or sequence of Axes
            The axes containing the plots, in state order.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from pykal.base import BaseKF
        >>> from pykal.utils.systemtype import SystemType
        >>> class IdentityKF(BaseKF):
        ...     def predict(self, x, P, F, t, u): return x, P
        ...     def update(self, x, P, y, H, t, u): return x, P
        >>> x_hist = pd.DataFrame([[0,1],[1,2],[2,3]], columns=['x','v'], index=[0,1,2])
        >>> cols = pd.MultiIndex.from_product([['x','v'], ['x','v']], names=['state_i','state_j'])
        >>> P_arr = np.tile(np.eye(2), (3,1,1)).reshape(3,-1)
        >>> P_hist = pd.DataFrame(P_arr, index=[0,1,2], columns=cols)
        >>> truth = pd.DataFrame([[0,1],[1,2],[2,3]], columns=['x','v'], index=[0,1,2])
        >>> meas  = pd.DataFrame([[0.1,1.1],[1.1,2.1],[2.1,3.1]], columns=['x','v'], index=[0,1,2])
        >>> kf = IdentityKF(f=lambda **kw: None, h=lambda **kw: None,
        ...                  Q=np.eye(2), R=np.eye(2),
        ...                  system_type=SystemType.DISCRETE_TIME_INVARIANT, dt=1.0)
        >>> axs = kf.plot_kf_predictions(x_hist, P_hist, x=truth, y=meas)
        >>> len(axs)
        2
        """
        # --- input validation ---
        if not isinstance(x_hist, pd.DataFrame):
            raise ValueError(f"x_hist must be a pd.DataFrame, got {type(x_hist)!r}")
        if not isinstance(P_hist, pd.DataFrame):
            raise ValueError(f"P_hist must be a pd.DataFrame, got {type(P_hist)!r}")
        # P_hist must have a MultiIndex of ('state_i','state_j')
        if not isinstance(P_hist.columns, pd.MultiIndex) or P_hist.columns.names != ['state_i','state_j']:
            raise ValueError("P_hist.columns must be a MultiIndex with names ['state_i','state_j']")
        # states match between x_hist and P_hist
        states = list(x_hist.columns)
        if set(P_hist.columns.get_level_values('state_i')) != set(states) \
            or set(P_hist.columns.get_level_values('state_j')) != set(states):
            raise ValueError("P_hist states must match x_hist.columns")
        # time index alignment
        if not x_hist.index.equals(P_hist.index):
            raise ValueError("Index of x_hist and P_hist must match (time axis)")

        for name, df in (('x', x), ('y', y), ('inputs_df', inputs_df)):
            if df is not None:
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"{name} must be a pd.DataFrame, got {type(df)!r}")
                if not df.index.equals(x_hist.index):
                    raise ValueError(f"Index of {name} must match x_hist.index")
                # drop any columns not in states
                extra = set(df.columns) - set(states)
                if extra:
                    raise ValueError(f"{name} has columns not in x_hist: {extra}")

        if not isinstance(include_sigma_uncertainty, bool):
            raise ValueError("include_sigma_uncertainty must be a bool")
        if not isinstance(n_sigma_bound, (int, float)):
            raise ValueError("n_sigma_bound must be a number")

        # prepare axes
        n_states = len(states)
        if ax is None:
            fig, axs = plt.subplots(n_states, 1, figsize=(10, 3 * n_states), sharex=True)
            if n_states == 1:
                axs = [axs]
        else:
            if isinstance(ax, plt.Axes):
                axs = [ax]
            else:
                axs = list(ax)
            if len(axs) != n_states:
                raise ValueError(f"Provided axes length {len(axs)} must equal number of states {n_states}")

        times = x_hist.index.to_numpy()

        # plot each state
        for i, state in enumerate(states):
            a = axs[i]
            a.plot(times, x_hist[state], linestyle='--', label=f"Estimate: {state}")
            if x is not None:
                a.plot(times, x[state], label=f"True: {state}")
            if y is not None:
                a.plot(times, y[state], marker='x', linestyle='None', label="Measurement")
            if inputs_df is not None:
                a.plot(times, inputs_df[state], label=f"Input: {state}")
            if include_sigma_uncertainty:
                var = P_hist[(state, state)].to_numpy()
                std = np.sqrt(var)
                a.fill_between(times,
                                x_hist[state] - n_sigma_bound * std,
                                x_hist[state] + n_sigma_bound * std,
                                alpha=0.3,
                                label=f"±{n_sigma_bound}σ")
            a.set_ylabel(state)
            a.grid(True)
            a.legend()

        axs[-1].set_xlabel("Time")
        return axs

    def plot_error(
        self,
        x_true_df: pd.DataFrame,
        x_est_df: pd.DataFrame,
        P_est: np.ndarray,
        ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None
    ) -> Union[plt.Axes, Sequence[plt.Axes]]:
        """
        Plot RMSE and NEES consistency metrics for Kalman filter results on provided axes
        or create new ones, and return the axes for further customization.

        Parameters
        ----------
        x_true_df : pd.DataFrame, shape (N, n_states)
            True state history; rows are time steps, columns are state names.
        x_est_df : pd.DataFrame, shape (N, n_states)
            Estimated state history with the same index and columns as `x_true_df`.
        P_est : np.ndarray, shape (N, n_states, n_states)
            Covariance history for each time step.
        ax : Axes or sequence of two Axes, optional
            Axes to plot on. If None, two new subplots are created (one for RMSE, one for NEES).
            If provided, must be a sequence of length 2.

        Returns
        -------
        axs : tuple of Axes
            (ax_rmse, ax_nees) containing the RMSE and NEES plots respectively.

        Raises
        ------
        ValueError
            If inputs are the wrong type or have incompatible shapes, or if `ax` is invalid.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from pykal.base import BaseKF
        >>> from pykal.utils.systemtype import SystemType
        >>> class IdentityKF(BaseKF):
        ...     def predict(self, x, P, F, t, u): return x, P
        ...     def update(self, x, P, y, H, t, u): return x, P
        >>> idx = [0,1,2]
        >>> df = pd.DataFrame([[1,2],[2,3],[3,4]], columns=['a','b'], index=idx)
        >>> P = np.tile(np.eye(2), (3,1,1))
        >>> kf = IdentityKF(f=lambda **kw: None, h=lambda **kw: None,
        ...                  Q=np.eye(2), R=np.eye(2),
        ...                  system_type=SystemType.DISCRETE_TIME_INVARIANT, dt=1.0)
        >>> ax_rmse, ax_nees = kf.plot_error(df, df, P)
        >>> ax_rmse.get_ylabel() == "RMSE"
        True
        >>> len(ax_nees.lines) > 0
        True
        """
        # --- input validation ---
        if not isinstance(x_true_df, pd.DataFrame):
            raise ValueError(f"x_true_df must be pd.DataFrame, got {type(x_true_df)!r}")
        if not isinstance(x_est_df, pd.DataFrame):
            raise ValueError(f"x_est_df must be pd.DataFrame, got {type(x_est_df)!r}")
        if not isinstance(P_est, np.ndarray):
            raise ValueError(f"P_est must be np.ndarray, got {type(P_est)!r}")

        N, n = x_true_df.shape
        if x_est_df.shape != (N, n):
            raise ValueError(f"x_est_df shape {x_est_df.shape} does not match x_true_df {x_true_df.shape}")
        if P_est.ndim != 3 or P_est.shape != (N, n, n):
            raise ValueError(f"P_est must have shape (N, n, n)=({N},{n},{n}), got {P_est.shape}")

        # --- prepare axes ---
        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        else:
            # accept a single Axes only if wrapped in a sequence of length 2
            if isinstance(ax, plt.Axes):
                raise ValueError("ax must be a sequence of two Axes, not a single Axes")
            axs = list(ax)
            if len(axs) != 2:
                raise ValueError(f"Provided axes length {len(axs)} must be 2")

        ax_rmse, ax_nees = axs

        # --- RMSE plot ---
        errors = x_est_df.values - x_true_df.values
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        ax_rmse.bar(x_true_df.columns, rmse)
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_title(f"State RMSE over {N} steps")
        ax_rmse.grid(True)

        # --- NEES plot ---
        nees = np.empty(N)
        for k in range(N):
            e = errors[k][:, None]
            nees[k] = (e.T @ np.linalg.inv(P_est[k]) @ e).item()
        lower = chi2.ppf(0.025, df=n)
        upper = chi2.ppf(0.975, df=n)
        ax_nees.plot(x_true_df.index.to_numpy(), nees, label="NEES")
        ax_nees.hlines([lower, upper], x_true_df.index[0], x_true_df.index[-1],
                    colors='r', linestyles='--', label="95% χ² bounds")
        ax_nees.set_xlabel("Time step")
        ax_nees.set_ylabel("NEES")
        ax_nees.set_title("NEES (Normalized Estimation Error Squared)")
        ax_nees.legend()
        ax_nees.grid(True)

        return ax_rmse, ax_nees
