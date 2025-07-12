import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import Union, Optional, Callable, Sequence, List
from pykal.system import System
from pykal.kf.kf import EKF, UKF


class Observer:
    def __init__(self, system: System) -> None:

        self.sys = system
        self.ekf = EKF(system)
        self.ukf = UKF(system)

    def compute_observability_matrix(
        self,
        *,
        x0: NDArray,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        override_system_f: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
        override_system_input_names: Union[Sequence[str], bool] = False,
        override_system_measurement_names: Union[Sequence[str], bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
        epsilon: float = 1e-5,
    ) -> NDArray:
        """
        Compute the empirical observability matrix by stacking finite differences
        of the output trajectories with respect to each perturbed state.

        Parameters
        ----------
        x0 : NDArray
            Initial state vector, shape (n, 1).
        t_span : tuple[float, float], optional
            Time interval [t0, tf] over which to simulate.
        dt : float, optional
            Time step used for simulations.
        t_vector : NDArray, optional
            Explicit time vector to override t_span/dt.
        override_system_u : Callable or bool, optional
            Control input function u(t), or False to use system default.
        override_system_h : Callable or bool, optional
            Measurement function h(x, u, t), or False to use system default.
        override_system_input_names : Sequence[str] or bool, optional
            Input names to override the system's input_names.
        override_system_measurement_names : Sequence[str] or bool, optional
            Measurement names to override the system's measurement_names.
        epsilon : float
            Perturbation magnitude for state deviations.

        Returns
        -------
        O : NDArray
            Empirical observability matrix of shape ((K * m), n)
            where K is the number of time steps and m is the number of measurements.
        """
        measurement_names = (
            self.sys.measurement_names
            if override_system_measurement_names is False
            else self.sys._validate_string_sequence(override_system_measurement_names)
        )
        input_names = (
            self.sys.input_names
            if override_system_input_names is False
            else self.sys._validate_string_sequence(override_system_input_names)
        )

        t_linspace = self.sys._standardize_time_input_to_linspace(t_vector, t_span, dt)
        n_states = len(self.sys.state_names)
        m_meas = len(measurement_names)
        K = len(t_linspace)

        # (K * m, n) matrix
        O = np.zeros((K * m_meas, n_states))

        for i in range(n_states):
            ei = np.zeros_like(x0, dtype=float)
            ei[i, 0] = epsilon
            x_plus = x0 + ei
            x_minus = x0 - ei

            X_plus, _ = self.sys.simulate_states(
                x0=x_plus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=override_system_state_names,
                override_system_input_names=override_system_input_names,
                override_system_Q=override_system_Q,
            )

            X_minus, _ = self.sys.simulate_states(
                x0=x_minus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=override_system_state_names,
                override_system_input_names=override_system_input_names,
                override_system_Q=override_system_Q,
            )

            Y_plus, _ = self.sys.simulate_measurements(
                X=X_plus,
                t_vector=t_linspace,
                override_system_h=override_system_h,
                override_system_u=override_system_u,
                override_system_measurement_names=measurement_names,
                override_system_input_names=input_names,
                override_system_R=override_system_R,
            )
            Y_minus, _ = self.sys.simulate_measurements(
                X=X_minus,
                t_vector=t_linspace,
                override_system_h=override_system_h,
                override_system_u=override_system_u,
                override_system_measurement_names=measurement_names,
                override_system_input_names=input_names,
                override_system_R=override_system_R,
            )

            delta_Y = (Y_plus - Y_minus) / (2 * epsilon)  # shape (m, K)

            # Reshape column-wise: [dy(t_0); dy(t_1); ...] ∈ ℝ^{K·m}
            O[:, i] = delta_Y.T.flatten()

        return O

    def compute_observability_grammian_from_observability_matrix(
        self,
        *,
        x0: NDArray,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_input_names: Union[Sequence[str], bool] = False,
        override_system_measurement_names: Union[Sequence[str], bool] = False,
        epsilon: float = 1e-5,
    ) -> NDArray:
        r"""
        Estimate the empirical observability Gramian using central finite differences
        of the output trajectories with respect to state perturbations.

        This captures the aggregate sensitivity of the output measurements to small
        perturbations in each state, over a time window [t0, tf].

        Empirical Gramian:
            .. math::
                W_O = \frac{1}{2\epsilon^2} \sum_{i=1}^n
                    \int_{t_0}^{t_f} (y_i^+(t) - y_i^-(t))(y_i^+(t) - y_i^-(t))^\top dt

        where :math:`y_i^+(t)` and :math:`y_i^-(t)` are the measurement trajectories
        from initial conditions :math:`x_0 \pm \epsilon e_i`.

        Parameters
        ----------
        h : Callable
            Measurement function h(x, u, t), returning (m, 1) array.
        x0 : NDArray
            Initial state vector, shape (n, 1).
        u : Callable
            Input function u(t) returning (m, 1).
        t_span : tuple[float, float]
            Time interval [t0, tf] over which to simulate.
        dt : float
            Time step used for simulations.
        epsilon : float, optional
            Perturbation magnitude for state deviations.

        Returns
        -------
        W : NDArray
            Empirical observability Gramian, shape (n, n).

        Examples
        --------
        >>> x0 = np.array([[1.0], [0.0]])
        >>> observer = Observer()  # Must provide .simulate_measurements
        >>> W.shape
        (2, 2)
        """

        O = self.compute_observability_matrix(
            x0=x0,
            t_span=t_span,
            dt=dt,
            t_vector=t_vector,
            override_system_u=override_system_u,
            override_system_h=override_system_h,
            override_system_input_names=override_system_input_names,
            override_system_measurement_names=override_system_measurement_names,
            epsilon=epsilon,
        )
        return O.T @ O

    def compute_CRB_from_observability_grammian(
        self, *, W: NDArray, output_sr: bool = False
    ) -> Union[pd.Series, NDArray]:
        """
        Compute the Cramér-Rao Bound (CRB) diagonal values for each state
        from the observability Gramian.

        Parameters
        ----------
        W : NDArray
            Empirical observability Gramian, shape (n, n)

        output_df : bool, optional
            If True, return the result as a pandas Series indexed by state names.

        Returns
        -------
        Union[pd.Series, NDArray]
            - If output_df is True: CRB diagonal values as a Series indexed by state names.
            - Otherwise: NumPy array of CRB diagonal values.
        """
        ridge = 1e-5  # regularization for numerical stability
        CRB = np.linalg.inv(W + ridge * np.eye(W.shape[0]))
        diag_vals = np.diag(CRB)

        if output_sr:
            return pd.Series(diag_vals, index=self.sys.state_names, name="CRB")
        else:
            return diag_vals

    def compute_state_observability_from_observability_grammian_nullspace(
        self,
        *,
        W: NDArray,
        tol: float = 1e-10,
        output_sr: bool = False,
    ) -> Union[pd.Series, NDArray]:
        """
        Compute per-state observability indices using the nullspace projection method.

        Parameters
        ----------
        W : NDArray
            Empirical observability Gramian, shape (n, n).
        tol : float
            Threshold below which eigenvalues are considered "zero" (nullspace).
        output_df : bool, optional
            If True, return results as a pandas Series with state names as index.

        Returns
        -------
        Union[pd.Series, NDArray]
            - If output_df is True: Series mapping state name → observability index ∈ [0, 1]
            - Otherwise: NumPy array of observability indices
        """
        n = W.shape[0]
        eigvals, eigvecs = np.linalg.eigh(W)

        nullspace_mask = eigvals < tol
        null_vectors = eigvecs[:, nullspace_mask]  # shape (n, r)

        scores = np.empty(n)
        for j in range(n):
            e_j = np.zeros((n, 1))
            e_j[j, 0] = 1.0
            proj = null_vectors.T @ e_j  # shape (r, 1)
            norm2 = np.sum(proj**2)
            scores[j] = float(1.0 - norm2)

        if output_sr:
            return pd.Series(
                scores, index=self.sys.state_names, name="Observability Index"
            )
        else:
            return scores

    def compute_state_observability_over_time_using_observability_matrix_based_method(
        self,
        *,
        x0: NDArray,
        observability_metric_of_states: Callable,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        divide_time_into_k_windows: Optional[int] = None,
        window_length_in_points: Optional[int] = None,
        window_length_in_time: Optional[float] = None,
        overlap_points_between_windows: Optional[int] = None,
        overlap_time_between_windows: Optional[float] = None,
        output_df: bool = False,
    ) -> Union[Union[list[NDArray], list[pd.Series]], pd.DataFrame]:
        """
        Compute per-state observability scores over time windows.

        Parameters
        ----------
        x0 : NDArray
            Initial state.
        observability_metric_of_states : Callable
            Function that computes per-state observability for a given time window.
            Must return a pd.Series when called with output_sr=True.
        ...
        output_df : bool
            If True, return a time-indexed DataFrame with observability scores.

        Returns
        -------
        Union[list[pd.Series], pd.DataFrame]
            - List of per-window Series if output_df=False
            - Time-indexed DataFrame if output_df=True
        """
        t_linspace = self.sys._standardize_time_input_to_linspace(t_vector, t_span, dt)
        list_of_t_windows = self.sys._standardize_make_time_windows(
            t_linspace,
            divide_time_into_k_windows=divide_time_into_k_windows,
            window_length_in_points=window_length_in_points,
            window_length_in_time=window_length_in_time,
            overlap_points_between_windows=overlap_points_between_windows,
            overlap_time_between_windows=overlap_time_between_windows,
        )

        scores_over_time = []
        times = []

        for window_t in list_of_t_windows:
            W = self.compute_observability_grammian_from_observability_matrix(
                x0=x0, t_vector=window_t
            )
            sr = observability_metric_of_states(W=W, output_sr=True)
            scores_over_time.extend([sr] * len(window_t))
            times.extend(window_t)

        if output_df:
            df = pd.DataFrame(scores_over_time, index=times)
            df.index.name = "time"
            df.columns.name = "state_observability"
            return df.sort_index()
        else:
            return scores_over_time, list_of_t_windows

    def compute_observability_of_states_from_P_matrix(
        self,
        *,
        P_series: pd.Series,
        t_eval: float,
        normalize: bool = True,
        clip: bool = True,
        override_system_state_names: Union[Sequence[str], bool] = False,
        eps: float = 1e-12,
        output_sr: bool = False,
    ) -> Union[NDArray, pd.Series]:
        """
        Compute stochastic observability scores per state from the posterior covariance
        matrix P(t), using the inverse diagonal method from Ramos et al. (2021).

        Parameters
        ----------
        P_series : pd.Series
            A time-indexed Series of (n x n) covariance matrices from a Kalman filter run.
        t_eval : float
            Time at which to evaluate the observability scores.
        normalize : bool, default True
            Whether to normalize scores so they sum to 1.
        clip : bool, default True
            Whether to clip scores to [0, 1] after normalization.
        override_system_state_names : Sequence[str] or bool, optional
            If provided, override the state names used for labeling.
        eps : float, default 1e-12
            Small constant to prevent division by zero.
        output_sr : bool, default False
            If True, return scores as a pandas Series; otherwise, return a NumPy ndarray.

        Returns
        -------
        Union[NDArray, pd.Series]
            Observability scores per state at `t_eval`.
        """
        if not isinstance(P_series, pd.Series):
            raise TypeError("P_series must be a pandas Series indexed by time")

        try:
            t_idx = P_series.index.get_indexer([t_eval], method="nearest")[0]
        except Exception as e:
            raise ValueError(f"Failed to locate nearest time to t={t_eval}: {e}")

        P = P_series.iloc[t_idx]

        if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("Each value in P_series must be a square NumPy array")

        diag = np.diag(P)
        inv_diag = 1.0 / (diag + eps)

        if normalize:
            inv_diag = inv_diag / np.sum(inv_diag)

        if clip:
            inv_diag = np.clip(inv_diag, 0.0, 1.0)

        state_names = (
            self.sys.state_names
            if override_system_state_names is False
            else self.sys._validate_string_sequence(override_system_state_names)
        )

        if len(inv_diag) != len(state_names):
            raise ValueError(
                "Length of P diagonal does not match number of state names"
            )

        if output_sr:
            return pd.Series(inv_diag, index=state_names, name=t_eval)
        else:
            return inv_diag

    def compute_state_observability_over_time_using_P_matrix_based_method(
        self,
        *,
        P_series: pd.Series,
        normalize: bool = True,
        clip: bool = True,
        eps: float = 1e-12,
        output_df: bool = False,
    ) -> Union[list[np.ndarray], pd.DataFrame]:
        """
        Compute stochastic observability indices for each state at each time point
        in a time-indexed posterior covariance Series using the inverse-diagonal method.

        Parameters
        ----------
        P_series : pd.Series
            A time-indexed Series of (n x n) covariance matrices from a Kalman filter run.
        normalize : bool, default True
            Whether to normalize scores so they sum to 1 at each time.
        clip : bool, default True
            Whether to clip each score to [0, 1].
        eps : float, default 1e-12
            Small constant to prevent division by zero.
        output_df : bool, default False
            If True, return a time-indexed DataFrame; otherwise return list of np.ndarray.

        Returns
        -------
        Union[list[np.ndarray], pd.DataFrame]
            - If output_df=False: list of arrays (length n) of per-state scores over time.
            - If output_df=True: DataFrame indexed by time, with one column per state.
        """
        scores_list = []
        times = []

        for t, P in P_series.items():
            diag = np.diag(P)
            inv_diag = 1.0 / (diag + eps)

            if normalize:
                inv_diag = inv_diag / np.sum(inv_diag)

            if clip:
                inv_diag = np.clip(inv_diag, 0.0, 1.0)

            scores_list.append(inv_diag)
            times.append(t)

        if output_df:
            df = pd.DataFrame(scores_list, index=times, columns=self.sys.state_names)
            df.index.name = "time"
            df.columns.name = "state_observability"
            return df

        return scores_list
