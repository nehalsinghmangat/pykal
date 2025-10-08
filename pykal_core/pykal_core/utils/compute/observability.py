import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union, Optional, Sequence
from pykal_core.blocks import DSBlock


class Observability:
    @classmethod
    def matrix(
        cls,
        sys: DSBlock,
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

        state_names = (
            sys.state_names
            if override_system_state_names is False
            else sys.safeio._validate_string_sequence(override_system_state_names)
        )
        measurement_names = (
            sys.output_names
            if override_system_measurement_names is False
            else sys.safeio._validate_string_sequence(override_system_measurement_names)
        )
        input_names = (
            sys.input_names
            if override_system_input_names is False
            else sys.safeio._validate_string_sequence(override_system_input_names)
        )

        t_linspace = sys.safeio._standardize_time_input_to_linspace(
            t_vector, t_span, dt
        )
        n_states = len(state_names)
        m_meas = len(measurement_names)
        K = len(t_linspace)

        # (K * m, n) matrix
        O = np.zeros((K * m_meas, n_states))

        for i in range(n_states):
            ei = np.zeros_like(x0, dtype=float)
            ei[i, 0] = epsilon
            x_plus = x0 + ei
            x_minus = x0 - ei

            X_plus, _ = sys.simulate_states(
                x0=x_plus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=state_names,
                override_system_input_names=input_names,
                override_system_Q=override_system_Q,
            )

            X_minus, _ = sys.simulate_states(
                x0=x_minus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=state_names,
                override_system_input_names=input_names,
                override_system_Q=override_system_Q,
            )

            Y_plus, _ = sys.simulate_measurements(
                X=X_plus,
                t_vector=t_linspace,
                override_system_h=override_system_h,
                override_system_u=override_system_u,
                override_system_measurement_names=measurement_names,
                override_system_input_names=input_names,
                override_system_R=override_system_R,
            )
            Y_minus, _ = sys.simulate_measurements(
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

    @classmethod
    def grammian(
        cls,
        sys: DSBlock,
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

        O = Observability.matrix(
            sys=sys,
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

    @classmethod
    def CRB(
        cls,
        sys: DSBlock,
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
        output_sr: bool = False,
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
        W = Observability.grammian(
            sys=sys,
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

        CRB = np.linalg.inv(W + ridge * np.eye(W.shape[0]))
        diag_vals = np.diag(CRB)

        if output_sr:
            return pd.Series(diag_vals, index=sys.state_names, name="CRB")
        else:
            return diag_vals

    @classmethod
    def of_states_from_grammian_nullspace(
        cls,
        sys: DSBlock,
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
            return pd.Series(scores, index=sys.state_names, name="Observability Index")
        else:
            return scores

    @classmethod
    def of_states_over_time_from_grammian_via_callable(
        cls,
        sys: DSBlock,
        *,
        x0: NDArray,
        grammian_to_scores_func: Callable,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        divide_time_into_k_windows: Optional[int] = None,
        window_length_in_points: Optional[int] = None,
        window_length_in_time: Optional[float] = None,
        overlap_points_between_windows: Optional[int] = None,
        overlap_time_between_windows: Optional[float] = None,
        output_df: bool = False,
        epsilon: float = 1e-5,
        **kwargs,
    ) -> Union[list[pd.Series], pd.DataFrame]:
        """
        Compute per-state observability scores over time windows using any method
        that maps a Gramian to a statewise score (e.g., nullspace, CRB).

        Parameters
        ----------
        x0 : NDArray
            Initial state.
        grammian_to_scores_func : Callable
            A function that accepts W (Gramian) and returns a score per state.
            Must accept `sys` and `W` as arguments.
        t_span, dt, t_vector : time config
            Either t_span + dt or explicit t_vector.
        window_* : windowing config
            Controls how time is split into windows.
        output_df : bool
            If True, return as time-indexed DataFrame.
        epsilon : float
            Perturbation magnitude for finite differences.
        kwargs : dict
            Additional arguments passed to `grammian_to_scores_func`.

        Returns
        -------
        Union[list[pd.Series], pd.DataFrame]
            List or time-indexed DataFrame of per-state scores over time.
        """
        t_linspace = sys.safeio._standardize_time_input_to_linspace(
            t_vector, t_span, dt
        )
        list_of_t_windows = sys.safeio._standardize_make_time_windows(
            t_linspace,
            divide_time_into_k_windows=divide_time_into_k_windows,
            window_length_in_points=window_length_in_points,
            window_length_in_time=window_length_in_time,
            overlap_points_between_windows=overlap_points_between_windows,
            overlap_time_between_windows=overlap_time_between_windows,
            dt=dt,
        )

        scores_over_time = []
        times = []

        for window_t in list_of_t_windows:
            W = cls.grammian(sys=sys, x0=x0, t_vector=window_t, epsilon=epsilon)
            sr = sys.safeio.smart_call(
                grammian_to_scores_func, sys=sys, W=W, output_sr=True, **kwargs
            )
            scores_over_time.append(sr)
            times.append(window_t[len(window_t) // 2])  # midpoint of window

        if output_df:
            df = pd.DataFrame(scores_over_time, index=times)
            df.index.name = "time"
            return df
        else:
            return scores_over_time

    @classmethod
    def of_states_from_P_projection_onto_canonical_basis(
        cls,
        sys: DSBlock,
        *,
        P0: NDArray,
        Pk: NDArray,
        eps: float = 1e-10,
        output_sr: bool = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
    ) -> Union[NDArray, pd.Series]:
        """
        Compute normalized state-wise uncertainty by projecting the posterior
        covariance onto canonical basis directions after whitening by P0.

        Each value reflects how much normalized uncertainty lies along a state axis.

        Returns
        -------
        NDArray or pd.Series
            Normalized posterior variances along canonical basis directions.
            Higher values indicate greater uncertainty.
        """

        try:
            L = cholesky(P0)
        except LinAlgError:
            L = cholesky(P0 + eps * np.eye(P0.shape[0]))

        P0_inv_sqrt = inv(L)
        Pk_dimensionless = P0_inv_sqrt @ Pk @ P0_inv_sqrt
        Pk_normalized = Pk_dimensionless / np.trace(Pk_dimensionless)

        statewise_variance_normalized = np.diag(Pk_normalized)

        if output_sr:
            state_names = sys.safeio._resolve_override_str_list(
                override_system_state_names, sys.state_names
            )
            return pd.Series(
                statewise_variance_normalized,
                index=state_names,
                name="Normalized Canonical Variance",
            )
        else:
            return statewise_variance_normalized

    @classmethod
    def of_states_from_P_spread_across_canonical_basis(
        cls,
        sys: DSBlock,
        *,
        P0: NDArray,
        Pk: NDArray,
        eps: float = 1e-10,
        output_sr: bool = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
    ) -> Union[NDArray, pd.Series]:
        """
        Compute the total normalized covariance spread across each canonical basis direction.

        This computes the row-sum of the trace-normalized, dimensionless covariance matrix:
            P̃ = inv(sqrt(P0)) @ Pk @ inv(sqrt(P0))
        Each row sum reflects how much total uncertainty is associated with that state,
        including its correlations with other states.

        Returns
        -------
        beta : NDArray or pd.Series
            Per-state uncertainty spread over canonical basis (higher = more correlated).
        """

        try:
            L = cholesky(P0)
        except LinAlgError:
            L = cholesky(P0 + eps * np.eye(P0.shape[0]))

        P0_inv_sqrt = inv(L)
        Pk_dimensionless = P0_inv_sqrt @ Pk @ P0_inv_sqrt
        Pk_normalized = Pk_dimensionless / np.trace(Pk_dimensionless)

        # Row sum of normalized dimensionless covariance matrix
        row_sums = np.sum(np.abs(Pk_normalized), axis=1)  # (n,) vector

        # Optional naming support
        if output_sr:
            state_names = sys.safeio._resolve_override_str_list(
                override_system_state_names, sys.state_names
            )
            return pd.Series(row_sums, index=state_names, name="Covariance Spread")
        else:
            return row_sums

    @classmethod
    def of_states_over_time_from_P_series_via_callable(
        cls,
        sys: DSBlock,
        *,
        P_series: pd.Series,
        method: Callable,
        eps: float = 1e-10,
        output_df: bool = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
        **kwargs,
    ) -> Union[pd.DataFrame, list[Union[NDArray, pd.Series]]]:
        """
        Evaluate a user-supplied `of_states_from_P_*` method over all time steps
        in a time-indexed Series of covariance matrices.

        Parameters
        ----------
        P_series : pd.Series
            A time-indexed Series of (n x n) covariance matrices.
        method : Callable
            A function like `of_states_from_P_projection_onto_canonical_basis`
            that accepts P0 and Pk and returns per-state observability.
        eps : float, default 1e-10
            Regularization for inversion/Cholesky.
        output_df : bool, default False
            If True, return a time-indexed DataFrame. Otherwise, return a list.
        override_system_state_names : Sequence[str] or bool, optional
            Custom state names for Series/DataFrame output.
        kwargs : dict
            Additional keyword arguments forwarded to the method.

        Returns
        -------
        Union[pd.DataFrame, list]
            - DataFrame if `output_df=True`: rows indexed by time, columns by state.
            - Otherwise, a list of arrays or Series (one per time).
        """
        P0 = P_series.iloc[0]

        results = []
        times = []

        for tk, Pk in P_series.items():
            result = sys.safeio.smart_call(
                method,
                P0=P0,
                Pk=Pk,
                eps=eps,
                sys=sys,
                output_sr=True,
                override_system_state_names=override_system_state_names,
                **kwargs,
            )
            results.append(result)
            times.append(tk)

        if output_df:
            df = pd.DataFrame(results, index=times)
            df.index.name = "time"
            return df
        else:
            return results
