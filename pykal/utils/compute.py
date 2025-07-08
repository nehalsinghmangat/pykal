import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Union, Optional, Callable, Sequence, List, Tuple
from pykal.utils.safeio import SafeIO as safeio
def error_pointwise(
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
    >>> from pykal.utils.utils_computation import Error
    >>> Error.compute_pointwise_error_metrics(truedf, estdf).round(2)
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


def error_nees(
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
    >>> import pandas as pd
    >>> from pykal.utils.utils_computation import Error
    >>> truedf = pd.DataFrame({'x0': [1.0], 'x1': [1.0]}, index=[0.0])
    >>> estdf = pd.DataFrame({'x0': [1.1], 'x1': [0.9]}, index=[0.0])
    >>> P_seq = pd.Series({0.0: 0.1 * np.eye(2)})
    >>> Error.compute_nees_only(truedf, estdf, P_seq).round(2)
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


def error_nll(
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
    >>> from pykal.utils.utils_computation import Error
    >>> truedf = pd.DataFrame({'x0': [1.0], 'x1': [2.0]}, index=[0.0])
    >>> estdf = pd.DataFrame({'x0': [1.1], 'x1': [1.9]}, index=[0.0])
    >>> P_seq = pd.Series({0.0: 0.1 * np.eye(2)})
    >>> Error.compute_nll(truedf, estdf, P_seq).round(2)
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


def error_mse_per_state(truedf: pd.DataFrame, estdf: pd.DataFrame) -> pd.Series:
    """
    Compute mean squared error per state over all time steps.

    Returns
    -------
    pd.Series
        MSE per state (averaged over time).

    Examples
    --------
    >>> import pandas as pd
    >>> from pykal.utils.utils_computation import Error
    >>> truedf = pd.DataFrame({'x0': [1.0, 2.0], 'x1': [2.0, 4.0]}, index=[0.0, 1.0])
    >>> estdf = pd.DataFrame({'x0': [1.1, 1.9], 'x1': [2.1, 3.8]}, index=[0.0, 1.0])
    >>> Error.compute_mse_per_state(truedf, estdf).round(4)
    x0    0.010
    x1    0.025
    dtype: float64
    """
    if not truedf.index.equals(estdf.index):
        raise ValueError("Indices must match.")

    diff = estdf[truedf.columns] - truedf
    return (diff**2).mean()





def compute_eigenvalues_and_eigenvectors_of_grammian(M: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Compute the eigenvalues and eigenvectors of a symmetric matrix.

    Parameters
    ----------
    M : NDArray
        A square symmetric matrix (e.g., observability Gramian)

    Returns
    -------
    Tuple[NDArray, NDArray]
        A tuple (eigvals, eigvecs) where:
        - eigvals is a 1D array of eigenvalues sorted in descending order.
        - eigvecs is a matrix whose columns are the corresponding eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eigh(M)  # stable for symmetric matrices
    idx = np.argsort(eigvals)[::-1]       # sort indices descending
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    return eigvals_sorted, eigvecs_sorted

def compute_condition_number_from_eigenvalues(eigvals: NDArray) -> float:
    """
    Compute the 2-norm condition number from a list of eigenvalues.

    Parameters
    ----------
    eigvals : NDArray
        A 1D array of real eigenvalues, typically from a symmetric matrix.

    Returns
    -------
    float
        Condition number (ratio of largest to smallest eigenvalue)
    """
    eigvals_sorted = np.sort(eigvals)[::-1]  # sort descending
    λ_max = eigvals_sorted[0]
    λ_min = eigvals_sorted[-1]
    if np.isclose(λ_min, 0.0):
        return np.inf
    return λ_max / λ_min


def plot_observability_directions_from_eigenpairs(
    eigvals: NDArray,
    eigvecs: NDArray,
    state_names: Optional[Sequence[str]] = None,
    projection_dims: tuple[int, int] = (0, 1),
    title: str = "Projected Observability Directions (2D)"
) -> None:
    """
    Plot eigenvectors of the observability Gramian projected into 2D to visualize
    most/least observable directions.

    Parameters
    ----------
    eigvals : NDArray
        1D array of eigenvalues (length n).
    eigvecs : NDArray
        2D array of eigenvectors (shape n x n), columns are eigenvectors.
    state_names : Sequence[str], optional
        Names of the states. Used to label axes. If None, index labels are used.
    projection_dims : tuple[int, int]
        Indices of the two state dimensions to project onto (e.g., (0,1)).
    title : str
        Title of the plot.
    """
    n = eigvals.shape[0]
    if eigvecs.shape != (n, n):
        raise ValueError(f"eigvecs must be of shape ({n}, {n}) to match eigvals")

    i, j = projection_dims
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"Invalid projection dimensions {projection_dims} for n={n}")

    # Sort by decreasing observability
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    # Normalize eigenvalue magnitudes
    scales = eigvals_sorted / np.max(eigvals_sorted)

    fig, ax = plt.subplots(figsize=(6, 6))
    origin = np.zeros(2)

    for k in range(n):
        vec_proj = eigvecs_sorted[[i, j], k] * scales[k]
        color = 'r' if k == 0 else ('b' if k == n - 1 else 'gray')
        label = "Most observable" if k == 0 else ("Least observable" if k == n - 1 else None)
        ax.quiver(*origin, *vec_proj, angles='xy', scale_units='xy', scale=1, color=color, label=label)

    xlabel = state_names[i] if state_names and len(state_names) > i else f"State {i}"
    ylabel = state_names[j] if state_names and len(state_names) > j else f"State {j}"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()
