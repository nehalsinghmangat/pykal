import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Matrix:
    @staticmethod
    def eigenvalues_and_eigenvectors_of_symmetric_matrix(
        M: NDArray,
    ) -> Tuple[NDArray, NDArray]:
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
        idx = np.argsort(eigvals)[::-1]  # sort indices descending
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]
        return eigvals_sorted, eigvecs_sorted

    @staticmethod
    def condition_number_from_eigenvalues(eigvals: NDArray) -> float:
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
