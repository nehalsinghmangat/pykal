import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple
from scipy.stats import multivariate_normal


class ParticleFilter:
    """
    Particle Filter for nonlinear/non-Gaussian state estimation.

    Based on Gordon, Salmond, & Smith (1993), "Novel approach to nonlinear/non-Gaussian
    Bayesian state estimation", IEE Proceedings F, Vol. 140, No. 2, pp. 107-113.

    The Particle Filter represents the posterior distribution p(x_k | y_{1:k}) using
    weighted particles, enabling estimation for arbitrary nonlinear dynamics and
    non-Gaussian noise.
    """

    @staticmethod
    def f(
        *,
        particles_weights: Tuple[NDArray, NDArray],
        yk: NDArray,
        f: Callable,
        f_params: Dict,
        h: Callable = None,
        h_params: Dict = None,
        likelihood: Callable = None,
        likelihood_params: Dict = None,
        Q: NDArray = None,
        R: NDArray = None,
        resample_threshold: float = 0.5,
        resample_method: str = 'systematic'
    ) -> Tuple[NDArray, NDArray]:
        """
        Perform one full predict-update-resample cycle of the Particle Filter.

        Parameters
        ----------
        particles_weights : Tuple[NDArray, NDArray]
            A tuple ``(particles, weights)`` containing:
                - ``particles`` : particle states, shape (N_p, n)
                - ``weights`` : normalized particle weights, shape (N_p,)
        yk : NDArray
            Measurement at time k, shape (m,)
        f : Callable
            Nonlinear dynamics function: x_{k+1} = f(x_k, **f_params)
            Should accept state as first argument
        f_params : Dict
            Additional parameters for dynamics function
        h : Callable, optional
            Nonlinear measurement function: y_k = h(x_k, **h_params)
            Required if likelihood is not provided
            Should accept state as first argument
        h_params : Dict, optional
            Additional parameters for measurement function
        likelihood : Callable, optional
            Custom likelihood function: p(y_k | x_k) = likelihood(yk, x_k, **likelihood_params)
            If provided, used instead of automatic Gaussian likelihood from h
            Should accept measurement and state as first two arguments
        likelihood_params : Dict, optional
            Additional parameters for likelihood function
        Q : NDArray, optional
            Process noise covariance, shape (n, n)
            Can be None if process noise is handled within the dynamics function f
        R : NDArray, optional
            Measurement noise covariance, shape (m, m)
            Required if h is provided (for Gaussian likelihood computation)
            Not needed if using custom likelihood function
        resample_threshold : float
            Effective sample size threshold (0-1) for triggering resampling
            ESS < threshold * N_p triggers resampling
        resample_method : str
            Resampling strategy: 'systematic', 'stratified', or 'residual'
            Default: 'systematic' (recommended for low variance)

        Returns
        -------
        particles_new : NDArray
            Updated particles, shape (N_p, n)
        weights_new : NDArray
            Updated normalized weights, shape (N_p,)

        Notes
        -----
        The Particle Filter approximates the posterior distribution using weighted samples:

        **Representation:**
            ``p(x_k | y_{1:k}) ≈ ∑_{i=1}^{N_p} w_k^(i) δ(x_k - x_k^(i))``

        **Algorithm Steps:**

        1. **Predict:** Propagate particles through dynamics with process noise
            ``x_k^(i) ~ p(x_k | x_{k-1}^(i))``  →  ``f(x_{k-1}^(i)) + w_k``

        2. **Update:** Compute likelihood weights
            ``w_k^(i) ∝ p(y_k | x_k^(i))``  →  evaluated via measurement model h()

        3. **Normalize:** Ensure weights sum to 1
            ``w_k^(i) = w_k^(i) / ∑_j w_k^(j)``

        4. **Resample:** If effective sample size (ESS) is low, resample to prevent degeneracy
            ``ESS = 1 / ∑_i (w_k^(i))²``

        **Numerical Stability:**
        - Weights computed in log-space to prevent underflow
        - Log-sum-exp trick for normalization
        - Small ridge term added to prevent zero weights

        **References:**
        Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993). Novel approach to
        nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F, 140(2), 107-113.
        """
        particles, weights = particles_weights
        N_p, n = particles.shape
        m = len(yk)

        # ===== PREDICT STEP =====
        # Propagate particles through dynamics with process noise
        particles_pred = np.zeros_like(particles)
        for i in range(N_p):
            # Propagate through dynamics
            x_pred = f(particles[i], **f_params)

            # Add process noise if Q is provided (otherwise assume f handles it)
            if Q is not None:
                process_noise = np.random.multivariate_normal(np.zeros(n), Q)
                particles_pred[i] = x_pred + process_noise
            else:
                particles_pred[i] = x_pred

        # ===== UPDATE STEP =====
        # Compute likelihood weights p(y_k | x_k^(i))
        log_weights = np.zeros(N_p)

        # Use custom likelihood if provided, otherwise compute Gaussian likelihood from h
        if likelihood is not None:
            # Custom likelihood function
            if likelihood_params is None:
                likelihood_params = {}

            for i in range(N_p):
                try:
                    weight = likelihood(yk, particles_pred[i], **likelihood_params)
                    log_weights[i] = np.log(weight + 1e-300)  # Prevent log(0)
                except:
                    log_weights[i] = -np.inf
        else:
            # Automatic Gaussian likelihood from measurement function h
            if h is None:
                raise ValueError("Either h or likelihood must be provided")
            if R is None:
                raise ValueError("R (measurement noise covariance) is required when using h (not using custom likelihood)")
            if h_params is None:
                h_params = {}

            for i in range(N_p):
                # Predicted measurement
                y_pred = h(particles_pred[i], **h_params)

                # Innovation (measurement residual)
                innovation = yk - y_pred

                # Log-likelihood: log p(y | x) = -0.5 * (y-h(x))^T R^{-1} (y-h(x)) + const
                try:
                    log_likelihood = multivariate_normal.logpdf(yk, mean=y_pred, cov=R)
                except:
                    # Fallback for numerical issues
                    try:
                        R_inv = np.linalg.inv(R)
                        log_likelihood = -0.5 * (innovation.T @ R_inv @ innovation)
                    except:
                        log_likelihood = -np.inf

                log_weights[i] = log_likelihood

        # Normalize weights (log-sum-exp trick for numerical stability)
        max_log_weight = np.max(log_weights)
        if not np.isfinite(max_log_weight):
            # All weights are -inf, reinitialize uniformly
            weights_new = np.ones(N_p) / N_p
        else:
            weights_new = np.exp(log_weights - max_log_weight)
            weights_new += 1e-300  # Prevent exact zeros
            weights_new /= np.sum(weights_new)

        # ===== RESAMPLE STEP =====
        # Check effective sample size
        ess = ParticleFilter.effective_sample_size(weights_new)
        ess_ratio = ess / N_p

        if ess_ratio < resample_threshold:
            # Resample particles to prevent degeneracy
            if resample_method == 'systematic':
                particles_new, weights_new = ParticleFilter._resample_systematic(
                    particles_pred, weights_new
                )
            elif resample_method == 'stratified':
                particles_new, weights_new = ParticleFilter._resample_stratified(
                    particles_pred, weights_new
                )
            elif resample_method == 'residual':
                particles_new, weights_new = ParticleFilter._resample_residual(
                    particles_pred, weights_new
                )
            else:
                raise ValueError(f"Unknown resampling method: {resample_method}")
        else:
            # No resampling needed
            particles_new = particles_pred

        return particles_new, weights_new

    @staticmethod
    def h(particles_weights: Tuple[NDArray, NDArray]) -> NDArray:
        """
        Compute the weighted mean state estimate from particles.

        Parameters
        ----------
        particles_weights : Tuple[NDArray, NDArray]
            Tuple (particles, weights)

        Returns
        -------
        xhat : NDArray
            Weighted mean state estimate, shape (n,)

        Notes
        -----
        The weighted mean is computed as:

        .. math::

            \\hat{x}_k = \\sum_{i=1}^{N_p} w_k^{(i)} x_k^{(i)}

        For multimodal distributions, the weighted mean may not be the best
        point estimate. Alternative estimators:
        - **Maximum a posteriori (MAP):** ``particles[np.argmax(weights)]``
        - **Median:** Component-wise weighted median

        Examples
        --------
        >>> import numpy as np
        >>> particles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> particles_weights = (particles, weights)
        >>> xhat = ParticleFilter.standard_h(particles_weights)
        >>> xhat
        array([2.4, 3.4])
        """
        particles, weights = particles_weights
        # Weighted mean estimate
        xhat = np.sum(weights[:, np.newaxis] * particles, axis=0)
        return xhat

    @staticmethod
    def effective_sample_size(weights: NDArray) -> float:
        """
        Compute the effective sample size (ESS) of the particle set.

        Parameters
        ----------
        weights : NDArray
            Normalized particle weights, shape (N_p,)

        Returns
        -------
        ess : float
            Effective sample size, range [1, N_p]

        Notes
        -----
        The effective sample size quantifies particle diversity:

        .. math::

            \\text{ESS} = \\frac{1}{\\sum_{i=1}^{N_p} (w_i)^2}

        **Interpretation:**
        - ``ESS ≈ N_p``: Weights are uniform (good diversity)
        - ``ESS ≈ 1``: One particle has all weight (degeneracy)

        **Resampling criterion:**
        Resample when ``ESS < threshold * N_p`` (typically threshold = 0.5)

        Examples
        --------
        >>> import numpy as np
        >>> # Uniform weights (maximum diversity)
        >>> weights = np.ones(100) / 100
        >>> ess = ParticleFilter.effective_sample_size(weights)
        >>> ess
        100.0
        >>> # Degenerate case (one particle has all weight)
        >>> weights = np.zeros(100)
        >>> weights[0] = 1.0
        >>> ess = ParticleFilter.effective_sample_size(weights)
        >>> ess
        1.0
        """
        return 1.0 / np.sum(weights**2)

    @staticmethod
    def _resample_systematic(
        particles: NDArray,
        weights: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Systematic resampling (low variance, recommended).

        Parameters
        ----------
        particles : NDArray
            Particle states, shape (N_p, n)
        weights : NDArray
            Normalized weights, shape (N_p,)

        Returns
        -------
        particles_new : NDArray
            Resampled particles, shape (N_p, n)
        weights_new : NDArray
            Uniform weights, shape (N_p,)

        Notes
        -----
        Systematic resampling uses deterministic spacing with a single random offset,
        resulting in low variance and computational efficiency O(N_p).
        """
        N_p = len(weights)

        # Single random offset for all samples
        positions = (np.arange(N_p) + np.random.random()) / N_p

        # Cumulative sum of weights
        cumsum = np.cumsum(weights)

        # Find indices via inverse CDF
        indices = np.searchsorted(cumsum, positions)

        # Resample particles
        particles_new = particles[indices]

        # Uniform weights after resampling
        weights_new = np.ones(N_p) / N_p

        return particles_new, weights_new

    @staticmethod
    def _resample_stratified(
        particles: NDArray,
        weights: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Stratified resampling (low variance).

        Parameters
        ----------
        particles : NDArray
            Particle states, shape (N_p, n)
        weights : NDArray
            Normalized weights, shape (N_p,)

        Returns
        -------
        particles_new : NDArray
            Resampled particles, shape (N_p, n)
        weights_new : NDArray
            Uniform weights, shape (N_p,)

        Notes
        -----
        Stratified resampling uses independent random offsets for each stratum,
        providing low variance similar to systematic resampling.
        """
        N_p = len(weights)

        # Independent random offset for each sample
        positions = (np.arange(N_p) + np.random.random(N_p)) / N_p

        # Cumulative sum of weights
        cumsum = np.cumsum(weights)

        # Find indices
        indices = np.searchsorted(cumsum, positions)

        # Resample particles
        particles_new = particles[indices]
        weights_new = np.ones(N_p) / N_p

        return particles_new, weights_new

    @staticmethod
    def _resample_residual(
        particles: NDArray,
        weights: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Residual resampling (minimum variance).

        Parameters
        ----------
        particles : NDArray
            Particle states, shape (N_p, n)
        weights : NDArray
            Normalized weights, shape (N_p,)

        Returns
        -------
        particles_new : NDArray
            Resampled particles, shape (N_p, n)
        weights_new : NDArray
            Uniform weights, shape (N_p,)

        Notes
        -----
        Residual resampling combines deterministic and stochastic sampling,
        achieving minimum variance among standard resampling methods.
        """
        N_p = len(weights)

        # Deterministic part: take floor(N_p * w_i) copies
        counts = np.floor(N_p * weights).astype(int)

        # Stochastic part: sample remainder
        residual = N_p * weights - counts
        residual_normalized = residual / np.sum(residual)

        # Multinomial sampling for remainder
        residual_counts = np.random.multinomial(N_p - np.sum(counts), residual_normalized)
        counts += residual_counts

        # Generate particle indices
        indices = np.repeat(np.arange(N_p), counts)

        # Resample particles
        particles_new = particles[indices]
        weights_new = np.ones(N_p) / N_p

        return particles_new, weights_new


# Alias for consistency with other algorithms (KF, UKF, etc.)
PF = ParticleFilter

# Module-level aliases for convenience
# Allows usage like: from pykal.algorithm_library.estimators import pf; pf.f(...)
f = ParticleFilter.f
h = ParticleFilter.h
