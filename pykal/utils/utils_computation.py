import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Callable, Optional
from scipy.linalg import svd, pinv

import numpy as np
from numpy.linalg import pinv
from numpy.typing import NDArray
from typing import Callable
from utils.compute_jacobian import (
    compute_empirical_jacobian,
)  # Adjust import to your path

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from numpy.typing import NDArray
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union
from utils.io_restrict import (
    validate_system_function,
    call_validated_function_with_args,
)


def compute_empirical_jacobian(
    func: Callable,
    x: NDArray,
    u: NDArray,
    t: float,
    epsilon: float = 1e-6,
    include_dt: bool = False,
) -> Union[
    tuple[NDArray, NDArray],  # if include_dt=False
    tuple[NDArray, NDArray, NDArray],  # if include_dt=True
]:
    """
    Compute empirical Jacobians of `func(x, u, t)` with respect to x, u, and optionally t.

    Parameters
    ----------
    func : Callable
        A validated function of the form f(x, u, t) → R^m.
    x : NDArray
        State vector of shape (n, 1).
    u : NDArray
        Input vector of shape (m, 1).
    t : float
        Scalar time.
    epsilon : float, optional
        Perturbation size.
    include_dt : bool, optional
        Whether to compute ∂f/∂t as well.

    Returns
    -------
    Jx : NDArray
        Jacobian ∂f/∂x of shape (m, n).
    Ju : NDArray
        Jacobian ∂f/∂u of shape (m, m_u).
    Jt : NDArray, optional
        Jacobian ∂f/∂t of shape (m, 1), returned if `include_dt=True`.

    Examples
    --------
    >>> import numpy as np
    >>> def f(x: NDArray, u:NDArray, t:float) -> NDArray:
    ...     return np.array([
    ...         [x[0, 0] + 2 * u[0, 0] + 0.1 * np.sin(t)],
    ...         [x[1, 0] ** 2 + u[0, 0] + t ** 2]
    ...     ])
    >>> x = np.array([[1.0], [2.0]])
    >>> u = np.array([[0.5]])
    >>> Jx, Ju, Jt = compute_empirical_jacobian(f, x, u, t=1.0, include_dt=True)
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

    func = validate_system_function(func)

    x = x.astype(float)
    u = u.astype(float)
    f0 = func(x, u, t)
    m = f0.shape[0]
    n_x = x.shape[0]
    n_u = u.shape[0]

    Jx = np.zeros((m, n_x))
    Ju = np.zeros((m, n_u))

    # ∂f/∂x
    for i in range(n_x):
        dx = np.zeros_like(x)
        dx[i, 0] = epsilon
        f_plus = call_validated_function_with_args(func, x=x + dx, u=u, t=t)
        f_minus = call_validated_function_with_args(func, x=x - dx, u=u, t=t)
        Jx[:, i : i + 1] = (f_plus - f_minus) / (2 * epsilon)

    # ∂f/∂u
    for j in range(n_u):
        du = np.zeros_like(u)
        du[j, 0] = epsilon
        f_plus = call_validated_function_with_args(func, x=x, u=u + du, t=t)
        f_minus = call_validated_function_with_args(func, x=x, u=u - du, t=t)
        Ju[:, j : j + 1] = (f_plus - f_minus) / (2 * epsilon)

    # ∂f/∂t
    if include_dt:
        f_plus = call_validated_function_with_args(func, x=x, u=u, t=t + epsilon)
        f_minus = call_validated_function_with_args(func, x=x, u=u, t=t - epsilon)
        Jt = (f_plus - f_minus) / (2 * epsilon)
        return Jx, Ju, Jt

    return Jx, Ju


def compute_empirical_observability_matrix_over_trajectory(
    X: pd.DataFrame,
    h: Callable,
    U: pd.DataFrame,
) -> pd.Series:
    """
    Compute the empirical observability matrix at each time step in a trajectory.

    At each time t, simulate output differences caused by perturbing the current
    state x(t) along canonical basis vectors, and return the best-fit Jacobian
    (observability matrix) ∂h/∂x using Y @ pinv(X).

    Parameters
    ----------
    X : pd.DataFrame
        State trajectory, shape (K, n). Row index = time.
    h : Callable
        Measurement function h(x, u, t) → R^m.
    U : pd.DataFrame
        Input trajectory, shape (K, m_u). Row index = time.

    Returns
    -------
    pd.Series
        A Series indexed by time, where each element is the observability matrix
        at that time (NDArray of shape (m, n)).

    Examples
    --------
    >>> def h(x, u, t): return x[0:1] + u[0:1] * np.sin(t)
    >>> t = np.array([0.0, 1.0])
    >>> x = np.array([[1.0, 1.1], [2.0, 2.1]]).T
    >>> u = np.array([[0.5, 0.5]]).T
    >>> X_df = pd.DataFrame(x, index=t, columns=["x0", "x1"])
    >>> U_df = pd.DataFrame(u, index=t, columns=["u0"])
    >>> O_series = compute_empirical_observability_matrix_over_trajectory(X_df, h, U_df)
    >>> O_series.iloc[0].shape
    (1, 2)
    """
    T = X.index
    n = X.shape[1]
    output_dim = h(
        X.iloc[0].values.reshape(-1, 1), U.iloc[0].values.reshape(-1, 1), T[0]
    ).shape[0]

    O_series = {}

    for t in T:
        x_t = X.loc[t].values.reshape(-1, 1)
        u_t = U.loc[t].values.reshape(-1, 1)
        y0 = h(x_t, u_t, t)
        Y = np.zeros((output_dim, n))

        for i in range(n):
            dx = np.zeros((n, 1))
            dx[i, 0] = 1.0
            y_pert = h(x_t + dx, u_t, t)
            Y[:, i] = (y_pert - y0).flatten()

        O_t = Y @ pinv(np.eye(n))  # or just O_t = Y
        O_series[t] = O_t

    return pd.Series(O_series, name="observability_matrix")


def compute_empirical_observability_gramian_over_trajectory(
    O_series: pd.Series,
) -> pd.Series:
    """
    Compute per-time observability Gramian contributions: W_k = O_k.T @ O_k.

    Parameters
    ----------
    O_series : pd.Series
        Time-indexed Series where each element is an (m, n) observability matrix

    Returns
    -------
    pd.Series
        Series indexed by time, where each value is an (n, n) matrix: W_k = O_kᵀ O_k

    Examples
    --------
    >>> def h(x, u, t): return x[0:1] + u[0:1] * np.sin(t)
    >>> t = np.array([0.0, 1.0])
    >>> x = np.array([[1.0, 1.1], [2.0, 2.1]]).T
    >>> u = np.array([[0.5, 0.5]]).T
    >>> X_df = pd.DataFrame(x, index=t, columns=["x0", "x1"])
    >>> U_df = pd.DataFrame(u, index=t, columns=["u0"])
    >>> O_series = compute_empirical_observability_matrix_over_trajectory(X_df, h, U_df)
    >>> W_series = compute_empirical_observability_gramian_over_trajectory(O_series)
    >>> W_series.iloc[0].shape
    (2, 2)
    >>> np.round(W_series.iloc[0], 2)
    array([[1., 0.],
           [0., 0.]])
    """

    return pd.Series({t: O.T @ O for t, O in O_series.items()})


def compute_crb_bound(P_seq: pd.Series) -> pd.Series:
    """
    Compute the Cramér–Rao Bound (CRB) proxy as the trace of each covariance matrix.

    Parameters
    ----------
    P_seq : pd.Series
        Series indexed by time, where each value is a covariance matrix (n × n)

    Returns
    -------
    pd.Series
        Series indexed by time, where each value is trace(P_k)

    Examples
    --------
    >>> t = np.array([0.0, 1.0, 2.0])
    >>> P0 = np.array([[1.0, 0.0], [0.0, 2.0]])
    >>> P1 = np.array([[0.5, 0.0], [0.0, 1.5]])
    >>> P2 = np.array([[0.2, 0.1], [0.1, 0.3]])
    >>> P_seq = pd.Series({0.0: P0, 1.0: P1, 2.0: P2})
    >>> crb = compute_crb_bound(P_seq)
    >>> crb.round(2)
    0.0    3.0
    1.0    2.0
    2.0    0.5
    dtype: float64
    """
    return pd.Series({t: np.trace(Pk) for t, Pk in P_seq.items()})


def compute_observability_singular_values(H_seq: list[NDArray]) -> NDArray:
    """
    Compute singular values of the stacked observability matrix.

    This helps diagnose how well the states are observable by measuring
    how "invertible" the observability matrix is. Small singular values
    indicate weak or unobservable directions.

    Parameters
    ----------
    H_seq : list of NDArray
        List of observability matrices H_k of shape (m_k, n)

    Returns
    -------
    NDArray
        Array of singular values sorted in descending order

    Examples
    --------
    >>> H0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> H1 = np.array([[1.0, 2.0]])
    >>> svals = compute_observability_singular_values([H0, H1])
    >>> np.round(svals, 3)
    array([2.288, 0.874])
    """
    return svd(np.vstack(H_seq), compute_uv=False)


def compute_fisher_information_matrix(
    H_seq: list[NDArray], R_seq: Optional[list[NDArray]] = None
) -> NDArray:
    """
    Compute the Fisher Information Matrix (FIM) over a trajectory:
        FIM = ∑ H_kᵀ · R_k⁻¹ · H_k

    If no R_seq is provided, assumes identity covariance.

    Parameters
    ----------
    H_seq : list of NDArray
        Sequence of Jacobian matrices H_k of shape (m_k, n)
    R_seq : list of NDArray, optional
        Sequence of measurement noise covariance matrices R_k of shape (m_k, m_k)

    Returns
    -------
    NDArray
        Fisher Information Matrix (n, n)

    Examples
    --------
    >>> H0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> H1 = np.array([[1.0, 2.0]])
    >>> R0 = np.eye(2)
    >>> R1 = np.array([[0.5]])
    >>> fim1 = compute_fisher_information_matrix([H0, H1])
    >>> np.round(fim1, 2)
    array([[2., 2.],
           [2., 2.]])
    >>> fim2 = compute_fisher_information_matrix([H0, H1], [R0, R1])
    >>> np.round(fim2, 2)
    array([[2., 4.],
           [4., 6.]])
    """
    FIM = np.zeros((H_seq[0].shape[1], H_seq[0].shape[1]))

    for i, H in enumerate(H_seq):
        R_inv = np.eye(H.shape[0]) if R_seq is None else np.linalg.inv(R_seq[i])
        FIM += H.T @ R_inv @ H

    return FIM


def compute_condition_number(O: NDArray, norm: int = 2) -> float:
    """Condition number of an observability matrix."""
    return np.linalg.cond(O, p=norm)


import pandas as pd
import numpy as np


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
