import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple


class UKF:
    """
    Unscented Kalman Filter (UKF) for nonlinear state estimation.

    Based on Julier & Uhlmann (2004), "Unscented Filtering and Nonlinear Estimation",
    Proceedings of the IEEE, Vol. 92, No. 3, pp. 401-422.

    The UKF uses the unscented transformation to propagate mean and covariance
    through nonlinear functions without requiring Jacobian calculations.
    """

    @staticmethod
    def _generate_sigma_points(
        x: NDArray,
        P: NDArray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ) -> Tuple[NDArray, NDArray]:
        """
        Generate sigma points for the unscented transformation.

        Parameters
        ----------
        x : NDArray
            State mean vector, shape (n,)
        P : NDArray
            State covariance matrix, shape (n, n)
        alpha : float
            Spread of sigma points around mean (typically 1e-4 to 1)
        beta : float
            Incorporate prior knowledge of distribution (2 is optimal for Gaussian)
        kappa : float
            Secondary scaling parameter (typically 0 or 3-n)

        Returns
        -------
        sigma_points : NDArray
            Sigma points, shape (2n+1, n)
        weights_mean : NDArray
            Weights for mean calculation, shape (2n+1,)
        weights_cov : NDArray
            Weights for covariance calculation, shape (2n+1,)
        """
        n = len(x)
        lambda_ = alpha**2 * (n + kappa) - n

        # Compute matrix square root of (n + lambda) * P
        try:
            # Use Cholesky decomposition (faster, numerically stable)
            L = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if Cholesky fails
            eigval, eigvec = np.linalg.eigh((n + lambda_) * P)
            eigval = np.maximum(eigval, 1e-10)  # Ensure positive eigenvalues
            L = eigvec @ np.diag(np.sqrt(eigval))

        # Generate sigma points
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = x  # Central sigma point

        for i in range(n):
            sigma_points[i + 1] = x + L[:, i]
            sigma_points[n + i + 1] = x - L[:, i]

        # Compute weights
        weights_mean = np.zeros(2*n + 1)
        weights_cov = np.zeros(2*n + 1)

        weights_mean[0] = lambda_ / (n + lambda_)
        weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

        for i in range(1, 2*n + 1):
            weights_mean[i] = 1 / (2 * (n + lambda_))
            weights_cov[i] = 1 / (2 * (n + lambda_))

        return sigma_points, weights_mean, weights_cov

    @staticmethod
    def _unscented_transform(
        sigma_points: NDArray,
        weights_mean: NDArray,
        weights_cov: NDArray,
        noise_cov: NDArray = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute mean and covariance from transformed sigma points.

        Parameters
        ----------
        sigma_points : NDArray
            Transformed sigma points, shape (2n+1, m)
        weights_mean : NDArray
            Weights for mean, shape (2n+1,)
        weights_cov : NDArray
            Weights for covariance, shape (2n+1,)
        noise_cov : NDArray, optional
            Additive noise covariance, shape (m, m)

        Returns
        -------
        mean : NDArray
            Weighted mean, shape (m,)
        cov : NDArray
            Weighted covariance, shape (m, m)
        """
        # Compute weighted mean
        mean = np.sum(weights_mean[:, np.newaxis] * sigma_points, axis=0)

        # Compute weighted covariance
        diff = sigma_points - mean
        cov = np.sum(weights_cov[:, np.newaxis, np.newaxis] *
                     (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]), axis=0)

        # Add noise covariance if provided
        if noise_cov is not None:
            cov = cov + noise_cov

        # Ensure symmetry
        cov = 0.5 * (cov + cov.T)

        return mean, cov

    @staticmethod
    def f(
        *,
        xhat_P: Tuple[NDArray, NDArray],
        yk: NDArray,
        f: Callable,
        f_params: Dict,
        h: Callable,
        h_params: Dict,
        Qk: NDArray,
        Rk: NDArray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ) -> Tuple[NDArray, NDArray]:
        """
        Perform one full predict-update step of the Unscented Kalman Filter.

        Parameters
        ----------
        xhat_P : Tuple[NDArray, NDArray]
            Tuple (x_hat_k, P_k) containing:
                - x_hat_k : current state estimate, shape (n,)
                - P_k : current state covariance, shape (n, n)
        yk : NDArray
            Measurement at time k, shape (m,)
        f : Callable
            Nonlinear state evolution function: x_{k+1} = f(x_k, **f_params)
            Should accept state as first argument
        f_params : Dict
            Additional parameters for state evolution function
        h : Callable
            Nonlinear measurement function: y_k = h(x_k, **h_params)
            Should accept state as first argument
        h_params : Dict
            Additional parameters for measurement function
        Qk : NDArray
            Process noise covariance, shape (n, n)
        Rk : NDArray
            Measurement noise covariance, shape (m, m)
        alpha : float
            UKF tuning parameter (spread of sigma points), typically 1e-4 to 1
        beta : float
            UKF tuning parameter (distribution shape), 2 is optimal for Gaussian
        kappa : float
            UKF tuning parameter (secondary scaling), typically 0 or 3-n

        Returns
        -------
        x_upd : NDArray
            Updated state estimate, shape (n,)
        P_upd : NDArray
            Updated state covariance, shape (n, n)

        Notes
        -----
        The UKF uses the unscented transformation to propagate the state estimate
        through nonlinear dynamics and measurement functions. Unlike the EKF,
        it does not require Jacobian calculations.

        **Predict Step:**
        1. Generate sigma points from current estimate
        2. Propagate sigma points through dynamics f()
        3. Compute predicted mean and covariance

        **Update Step:**
        1. Propagate predicted sigma points through measurement h()
        2. Compute predicted measurement mean and covariance
        3. Compute cross-covariance between state and measurement
        4. Compute Kalman gain and update estimate

        **References:**
        Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear
        estimation. Proceedings of the IEEE, 92(3), 401-422.
        """
        x_hat, P = xhat_P
        n = len(x_hat)

        # ===== PREDICT STEP =====

        # Generate sigma points from current estimate
        sigma_points, w_mean, w_cov = UKF._generate_sigma_points(
            x_hat, P, alpha, beta, kappa
        )

        # Propagate sigma points through dynamics
        sigma_pred = np.zeros_like(sigma_points)
        for i in range(2*n + 1):
            sigma_pred[i] = f(sigma_points[i], **f_params)

        # Compute predicted mean and covariance
        x_pred, P_pred = UKF._unscented_transform(sigma_pred, w_mean, w_cov, Qk)

        # ===== UPDATE STEP =====

        # Generate new sigma points from predicted state
        sigma_points_pred, w_mean, w_cov = UKF._generate_sigma_points(
            x_pred, P_pred, alpha, beta, kappa
        )

        # Propagate sigma points through measurement function
        m = len(yk)  # Measurement dimension
        gamma_pred = np.zeros((2*n + 1, m))
        for i in range(2*n + 1):
            gamma_pred[i] = h(sigma_points_pred[i], **h_params)

        # Compute predicted measurement mean and covariance
        y_pred, S = UKF._unscented_transform(gamma_pred, w_mean, w_cov, Rk)

        # Compute cross-covariance between state and measurement
        diff_x = sigma_points_pred - x_pred
        diff_y = gamma_pred - y_pred
        Pxy = np.sum(w_cov[:, np.newaxis, np.newaxis] *
                     (diff_x[:, :, np.newaxis] @ diff_y[:, np.newaxis, :]), axis=0)

        # Compute Kalman gain
        ridge = 1e-9 * np.eye(S.shape[0])
        try:
            S_inv = np.linalg.inv(S + ridge)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S + ridge)

        K = Pxy @ S_inv

        # Update state and covariance
        innovation = yk - y_pred
        x_upd = x_pred + K @ innovation
        P_upd = P_pred - K @ S @ K.T

        # Ensure covariance symmetry
        P_upd = 0.5 * (P_upd + P_upd.T)

        return x_upd, P_upd

    @staticmethod
    def h(xhat_P: Tuple[NDArray, NDArray]) -> NDArray:
        """
        Extract current state estimate.

        Parameters
        ----------
        xhat_P : Tuple[NDArray, NDArray]
            State estimate and covariance tuple

        Returns
        -------
        x_hat : NDArray
            Current state estimate
        """
        return xhat_P[0]


# Module-level aliases for convenience
# Allows usage like: from pykal.algorithm_library.estimators import ukf; ukf.f(...)
f = UKF.f
h = UKF.h
