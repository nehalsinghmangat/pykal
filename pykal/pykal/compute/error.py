import pandas as pd
import numpy as np


class Error:
    @staticmethod
    def pointwise(
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
        >>> from pykal_core.utils.utils_computation import Error
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

    @staticmethod
    def nees(
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
        >>> from pykal_core.utils.utils_computation import Error
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

    @staticmethod
    def nll(
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
        >>> from pykal_core.utils.utils_computation import Error
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

    @staticmethod
    def mse_per_state(truedf: pd.DataFrame, estdf: pd.DataFrame) -> pd.Series:
        """
        Compute mean squared error per state over all time steps.

        Returns
        -------
        pd.Series
            MSE per state (averaged over time).

        Examples
        --------
        >>> import pandas as pd
        >>> from pykal_core.utils.utils_computation import Error
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
