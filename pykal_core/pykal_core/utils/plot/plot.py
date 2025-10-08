import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Optional, Sequence


class Plot:
    @staticmethod
    def plot_observability_directions_from_eigenpairs(
        eigvals: NDArray,
        eigvecs: NDArray,
        state_names: Optional[Sequence[str]] = None,
        projection_dims: tuple[int, int] = (0, 1),
        title: str = "Projected Observability Directions (2D)",
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
            raise ValueError(
                f"Invalid projection dimensions {projection_dims} for n={n}"
            )

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
            color = "r" if k == 0 else ("b" if k == n - 1 else "gray")
            label = (
                "Most observable"
                if k == 0
                else ("Least observable" if k == n - 1 else None)
            )
            ax.quiver(
                *origin,
                *vec_proj,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                label=label,
            )

        xlabel = (
            state_names[i] if state_names and len(state_names) > i else f"State {i}"
        )
        ylabel = (
            state_names[j] if state_names and len(state_names) > j else f"State {j}"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.set_aspect("equal")
        ax.legend()
        plt.tight_layout()
        plt.show()
