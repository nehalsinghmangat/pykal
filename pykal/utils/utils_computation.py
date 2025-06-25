from os import stat
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Callable, Optional, Union
from scipy.linalg import svd, pinv
import numpy as np
from numpy.linalg import pinv
from .utils_safeio import SafeIO as safeio


class ComputationIO:
    pass


class Differentiation:
    @staticmethod
    def compute_empirical_jacobian(
        func: Callable,
        xk: NDArray,
        uk: NDArray,
        tk: float,
        epsilon: float = 1e-6,
        include_dt: bool = False,
    ) -> Union[tuple[NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
        """
        Compute empirical Jacobians of a function f(x, u, t) with respect to state, input, and optionally time,
        using central finite differences.

        This is typically used for estimating the Jacobians of a nonlinear **dynamics function** in Kalman filters:
            f : Rⁿ × Rᵐ × R → Rⁿ

        Parameters
        ----------
        func : Callable
            A validated dynamics function of the form f(x, u, t) → Rⁿ, returning a (n, 1) ndarray.
        xk : NDArray
            State vector at time t, of shape (n, 1).
        uk : NDArray
            Input vector at time t, of shape (m, 1).
        tk : float
            Scalar time.
        epsilon : float, optional
            Perturbation size for finite differences. Default is 1e-6.
        include_dt : bool, optional
            Whether to compute the time derivative ∂f/∂t. Default is False.

        Returns
        -------
        Jx : NDArray
            Jacobian ∂f/∂x of shape (n, n).
        Ju : NDArray
            Jacobian ∂f/∂u of shape (n, m).
        Jt : NDArray, optional
            Jacobian ∂f/∂t of shape (n, 1), returned only if `include_dt=True`.

        Examples
        --------
        >>> import numpy as np
        >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return np.array([
        ...         [x[0, 0] + 2 * u[0, 0] + 0.1 * np.sin(t)],
        ...         [x[1, 0] ** 2 + u[0, 0] + t ** 2]
        ...     ])
        >>> x = np.array([[1.0], [2.0]])
        >>> u = np.array([[0.5]])
        >>> Jx, Ju, Jt = Differentiation.compute_empirical_jacobian(f, x, u, tk=1.0, include_dt=True)
        >>> np.round(Jx, 3)
        array([[1., 0.],
               [0., 4.]])
        >>> np.round(Ju, 3)
        array([[2.],
               [1.]])
        >>> np.round(Jt, 3)
        array([[0.054],
               [2.   ]])
        """

        n_states = xk.shape[0]
        n_inputs = uk.shape[0]

        Jx = np.zeros((n_states, n_states))
        Ju = np.zeros((n_states, n_inputs))
        Jt = np.zeros((n_states, 1))

        # ∂f/∂x
        for i in range(n_states):
            dx = np.zeros_like(xk)
            dx[i, 0] = epsilon
            f_plus = safeio.call_validated_function_with_args(
                func, x=xk + dx, u=uk, t=tk, expected_shape=(n_states, 1)
            )
            f_minus = safeio.call_validated_function_with_args(
                func, x=xk - dx, u=uk, t=tk, expected_shape=(n_states, 1)
            )
            Jx[:, i : i + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂u
        for j in range(n_inputs):
            du = np.zeros_like(uk)
            du[j, 0] = epsilon
            f_plus = safeio.call_validated_function_with_args(
                func, x=xk, u=uk + du, t=tk, expected_shape=(n_states, 1)
            )
            f_minus = safeio.call_validated_function_with_args(
                func, x=xk, u=uk - du, t=tk, expected_shape=(n_states, 1)
            )
            Ju[:, j : j + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂t
        if include_dt:
            f_plus = safeio.call_validated_function_with_args(
                func, x=xk, u=uk, t=tk + epsilon, expected_shape=(n_states, 1)
            )
            f_minus = safeio.call_validated_function_with_args(
                func, x=xk, u=uk, t=tk - epsilon, expected_shape=(n_states, 1)
            )
            Jt = (f_plus - f_minus) / (2 * epsilon)
            return Jx, Ju, Jt

        return Jx, Ju


class Observability:
    @staticmethod
    def build_stochastic_observability_beta_from_series(
        P_series: pd.Series,
    ) -> Callable[[float], NDArray]:
        """
        Returns a callable beta(t) using the stochastic observability method,
        based on a Series of posterior covariance matrices indexed by time.

        Parameters
        ----------
        P_series : pd.Series
            Series where the index is time (float) and values are covariance
            matrices P_k ∈ R^{n x n}.

        Returns
        -------
        beta_fn : Callable[[float], NDArray]
            Function returning observability weights beta(t_k) ∈ R^{n x 1}.
        """
        time_index = P_series.index.to_numpy()

        def beta(t: float) -> NDArray:
            nearest_t = time_index[np.argmin(np.abs(time_index - t))]
            Pk = P_series.loc[nearest_t]

            if not isinstance(Pk, np.ndarray) or Pk.ndim != 2:
                raise ValueError(f"P({nearest_t}) must be a 2D NDArray")

            diag = np.diag(Pk).reshape(-1, 1)
            λ_max = np.max(np.linalg.eigvalsh(Pk))
            if λ_max <= 0:
                raise ValueError(f"P({nearest_t}) is not positive semi-definite")

            return 1.0 - diag / λ_max

        return beta


class Error:
    @staticmethod
    def compute_pointwise_error_metrics(
        truedf: pd.DataFrame,
        estdf: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute pointwise RMSE, MAE, and MaxErr between true and estimated states.

        Parameters
        ----------
        truedf : pd.DataFrame
            Ground truth state values.
        estdf : pd.DataFrame
            Estimated state values.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['RMSE', 'MAE', 'MaxErr'].

        Examples
        --------
        >>> truedf = pd.DataFrame({'x0': [1.0, 2.0], 'x1': [1.0, 3.0]}, index=[0.0, 1.0])
        >>> estdf = pd.DataFrame({'x0': [1.1, 1.9], 'x1': [0.9, 3.2]}, index=[0.0, 1.0])
        >>> compute_pointwise_error_metrics(truedf, estdf).round(2)
              RMSE  MAE  MaxErr
        0.0  0.10  0.10     0.1
        1.0  0.16  0.15     0.2
        """
        if not truedf.index.equals(estdf.index):
            raise ValueError("Indices of true and estimated DataFrames must match.")

        errors = (estdf[truedf.columns] - truedf).values  # shape (T, n)

        rmse = np.sqrt(np.mean(errors**2, axis=1))
        mae = np.mean(np.abs(errors), axis=1)
        maxerr = np.max(np.abs(errors), axis=1)

        return pd.DataFrame(
            {"RMSE": rmse, "MAE": mae, "MaxErr": maxerr}, index=truedf.index
        )

    @staticmethod
    def compute_nees_only(
        truedf: pd.DataFrame,
        estdf: pd.DataFrame,
        P_seq: pd.Series,
        tol: float = 1e-9,
    ) -> pd.Series:
        """
        Compute the Normalized Estimation Error Squared (NEES) over time.

        Parameters
        ----------
        truedf : pd.DataFrame
            True state values indexed by time.
        estdf : pd.DataFrame
            Estimated state values indexed by time.
        P_seq : pd.Series of np.ndarray
            Covariance matrices (n × n), one per time step, indexed by time.
        tol : float, optional
            Small diagonal regularization added to P_k for numerical stability.

        Returns
        -------
        pd.Series
            NEES values indexed by time.

        Raises
        ------
        ValueError
            If indices do not match or if P_seq entries are not 2D arrays.

        Examples
        --------
        >>> truedf = pd.DataFrame({'x0': [1.0], 'x1': [1.0]}, index=[0.0])
        >>> estdf = pd.DataFrame({'x0': [1.1], 'x1': [0.9]}, index=[0.0])
        >>> P_seq = pd.Series({0.0: 0.1 * np.eye(2)})
        >>> compute_nees_only(truedf, estdf, P_seq).round(2)
        0.0    0.2
        Name: NEES, dtype: float64
        """
        if not isinstance(P_seq, pd.Series):
            raise TypeError("P_seq must be a pandas Series of covariance matrices.")

        if not truedf.index.equals(estdf.index) or not truedf.index.equals(P_seq.index):
            raise ValueError("Indices of truedf, estdf, and P_seq must all match.")

        nees_values = []

        for t in truedf.index:
            x_true = truedf.loc[t].values.reshape(-1, 1)
            x_est = estdf.loc[t].values.reshape(-1, 1)
            e_k = x_est - x_true

            P_k = P_seq.loc[t]
            if not isinstance(P_k, np.ndarray) or P_k.ndim != 2:
                raise ValueError(f"P_seq[{t}] must be a 2D NumPy array.")

            P_k_reg = P_k + tol * np.eye(P_k.shape[0])

            try:
                nees_k = float(e_k.T @ np.linalg.inv(P_k_reg) @ e_k)
            except np.linalg.LinAlgError:
                nees_k = np.nan

            nees_values.append(nees_k)

        return pd.Series(nees_values, index=truedf.index, name="NEES")

    @staticmethod
    def compute_nll(
        truedf: pd.DataFrame,
        estdf: pd.DataFrame,
        P_seq: pd.Series,
        tol: float = 1e-9,
    ) -> pd.Series:
        """
        Compute the Negative Log-Likelihood (NLL) over time under Gaussian assumption.

        Parameters
        ----------
        truedf : pd.DataFrame
            True state values.
        estdf : pd.DataFrame
            Estimated state values.
        P_seq : pd.Series of np.ndarray
            Covariance matrices per timestep.
        tol : float, optional
            Diagonal regularization.

        Returns
        -------
        pd.Series
            Negative log-likelihood values indexed by time.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> truedf = pd.DataFrame({'x0': [1.0], 'x1': [2.0]}, index=[0.0])
        >>> estdf = pd.DataFrame({'x0': [1.1], 'x1': [1.9]}, index=[0.0])
        >>> P_seq = pd.Series({0.0: 0.1 * np.eye(2)})
        >>> compute_nll(truedf, estdf, P_seq).round(2)
        0.0    -0.36
        Name: NLL, dtype: float64
        """
        if not truedf.index.equals(estdf.index) or not truedf.index.equals(P_seq.index):
            raise ValueError("Indices must match for truedf, estdf, and P_seq.")

        nll_vals = []

        for t in truedf.index:
            x_true = truedf.loc[t].values.reshape(-1, 1)
            x_est = estdf.loc[t].values.reshape(-1, 1)
            e_k = x_est - x_true
            P_k = P_seq.loc[t] + tol * np.eye(len(e_k))

            try:
                inv_P = np.linalg.inv(P_k)
                nees_k = float(e_k.T @ inv_P @ e_k)
                logdet = np.linalg.slogdet(2 * np.pi * P_k)[1]
                nll_k = 0.5 * (logdet + nees_k)
            except np.linalg.LinAlgError:
                nll_k = np.nan

            nll_vals.append(nll_k)

        return pd.Series(nll_vals, index=truedf.index, name="NLL")

    @staticmethod
    def compute_mse_per_state(truedf: pd.DataFrame, estdf: pd.DataFrame) -> pd.Series:
        """
        Compute mean squared error per state over all time steps.

        Returns
        -------
        pd.Series
            MSE per state (averaged over time).

        Examples
        --------
        >>> import pandas as pd
        >>> truedf = pd.DataFrame({'x0': [1.0, 2.0], 'x1': [2.0, 4.0]}, index=[0.0, 1.0])
        >>> estdf = pd.DataFrame({'x0': [1.1, 1.9], 'x1': [2.1, 3.8]}, index=[0.0, 1.0])
        >>> compute_mse_per_state(truedf, estdf).round(4)
        x0    0.010
        x1    0.025
        dtype: float64
        """
        if not truedf.index.equals(estdf.index):
            raise ValueError("Indices must match.")

        diff = estdf[truedf.columns] - truedf
        return (diff**2).mean()
