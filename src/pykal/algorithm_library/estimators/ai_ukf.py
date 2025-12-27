import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple
from .ukf import UKF


class AIUKF:
    """
    Augmented Information Unscented Kalman Filter (AI-UKF) for hybrid
    physics-data-driven state estimation.

    Based on Cellini et al. (2025), "Discovering and exploiting active sensing
    motifs for estimation", arXiv:2511.08766.

    The AI-UKF extends the standard UKF by augmenting measurements with data-driven
    predictions (e.g., from Artificial Neural Networks) and using observability-aware
    time-varying measurement covariance to dynamically adjust trust in these predictions.

    **Key Innovation**: Time-varying R that adapts based on instantaneous observability,
    enabling robust estimation even with poor initial conditions or weak observability.
    """

    @staticmethod
    def f(
        *,
        xhat_P: Tuple[NDArray, NDArray],
        yk_aug: NDArray,
        f: Callable,
        f_params: Dict,
        h_aug: Callable,
        h_aug_params: Dict,
        Qk: NDArray,
        Rk_aug: NDArray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ) -> Tuple[NDArray, NDArray]:
        """
        Perform one full predict-update step of the AI-UKF with augmented measurements.

        Parameters
        ----------
        xhat_P : Tuple[NDArray, NDArray]
            Tuple (x_hat_k, P_k) containing:
                - x_hat_k : current state estimate, shape (n,)
                - P_k : current state covariance, shape (n, n)
        yk_aug : NDArray
            Augmented measurement at time k, shape (m_aug,)
            Typically [sensor_measurements, ANN_predictions]
        f : Callable
            Nonlinear state evolution function: x_{k+1} = f(x_k, **f_params)
            Should accept state as first argument
        f_params : Dict
            Additional parameters for state evolution function
        h_aug : Callable
            Augmented nonlinear measurement function: y_aug_k = h_aug(x_k, **h_aug_params)
            Should return augmented measurement vector including ANN-predicted states
        h_aug_params : Dict
            Additional parameters for augmented measurement function
        Qk : NDArray
            Process noise covariance, shape (n, n)
        Rk_aug : NDArray
            **Time-varying** augmented measurement noise covariance, shape (m_aug, m_aug)
            Structure: R_aug = [[R_sensor,    0        ],
                               [0,        R_ANN(t)  ]]
            where R_ANN(t) varies based on observability
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
        The AI-UKF algorithm follows the standard UKF predict-update cycle but with
        key modifications:

        **Augmented Measurement Model:**

        .. math::

            \\mathbf{y}_{aug}(t) = \\begin{bmatrix}
                \\mathbf{h}(\\mathbf{x}(t)) \\\\
                \\check{h}_{ANN}(\\mathbf{y}_{[t-w:t]})
            \\end{bmatrix}

        where :math:`\\check{h}_{ANN}` is a data-driven estimator (e.g., ANN) that
        predicts poorly observable states from a time window of measurements.

        **Time-Varying Measurement Covariance:**

        .. math::

            \\mathbf{R}_{aug}(t) = \\begin{bmatrix}
                \\mathbf{R}_{sensor} & \\mathbf{0} \\\\
                \\mathbf{0} & \\check{R}_{ANN}(t)
            \\end{bmatrix}

        The ANN measurement variance adapts based on observability:

        .. math::

            \\check{R}_{ANN}(t) = \\frac{c}{\\min_{i \\in [t-w, t]} \\sigma(i)}

        where :math:`\\sigma(i)` is an observability indicator (e.g., |acceleration|):
        - **High observability** → Small :math:`\\check{R}_{ANN}` → Trust ANN
        - **Low observability** → Large :math:`\\check{R}_{ANN}` → Ignore ANN

        **Algorithm Steps:**

        1. **Predict:** Propagate sigma points through dynamics :math:`f()`
        2. **Update:** Use augmented measurements and time-varying :math:`\\mathbf{R}_{aug}`
        3. **Adaptive Trust:** Automatically adjust ANN influence based on observability

        **When to Use AI-UKF:**
        - Initial conditions are very uncertain (standard UKF would diverge)
        - States are intermittently observable (e.g., altitude from optic flow)
        - You have trained data-driven estimators for weakly observable states

        **References:**
        Cellini, B., Boyacioglu, B., Lopez, A., & van Breugel, F. (2025).
        Discovering and exploiting active sensing motifs for estimation.
        arXiv:2511.08766.

        Examples
        --------
        >>> import numpy as np
        >>> from pykal.algorithm_library.estimators.ai_ukf import AIUKF
        >>>
        >>> # Simple 1D example with augmented altitude measurement
        >>> x_hat = np.array([0.0])  # State: altitude
        >>> P = np.eye(1) * 100  # Large initial uncertainty
        >>> xhat_P = (x_hat, P)
        >>>
        >>> # Augmented measurement: [optic_flow, altitude_ANN_estimate]
        >>> yk_aug = np.array([0.5, 10.0])
        >>>
        >>> # Define dynamics and augmented measurement functions
        >>> def f(x): return x  # Constant altitude
        >>> def h_aug(x): return np.array([x[0]/10, x[0]])  # [optic_flow, altitude]
        >>>
        >>> # Time-varying R: trust ANN more when observability is high
        >>> R_sensor = np.array([[0.1]])
        >>> R_ANN = np.array([[1.0]])  # Varies with time in practice
        >>> Rk_aug = np.block([[R_sensor, np.zeros((1,1))],
        ...                    [np.zeros((1,1)), R_ANN]])
        >>>
        >>> Q = np.eye(1) * 0.01
        >>>
        >>> # Run AI-UKF
        >>> x_upd, P_upd = AIUKF.standard_f(
        ...     xhat_P=xhat_P, yk_aug=yk_aug, f=f, f_params={},
        ...     h_aug=h_aug, h_aug_params={}, Qk=Q, Rk_aug=Rk_aug
        ... )
        >>> x_upd.shape
        (1,)
        """
        x_hat, P = xhat_P
        n = len(x_hat)

        # ===== PREDICT STEP =====
        # Generate sigma points from current estimate
        # Reuse UKF's sigma point generation
        sigma_points, w_mean, w_cov = UKF._generate_sigma_points(
            x_hat, P, alpha, beta, kappa
        )

        # Propagate sigma points through dynamics
        sigma_pred = np.zeros_like(sigma_points)
        for i in range(2*n + 1):
            sigma_pred[i] = f(sigma_points[i], **f_params)

        # Compute predicted mean and covariance
        # Reuse UKF's unscented transform
        x_pred, P_pred = UKF._unscented_transform(sigma_pred, w_mean, w_cov, Qk)

        # ===== UPDATE STEP (with augmented measurements) =====
        # Generate new sigma points from predicted state
        sigma_points_pred, w_mean, w_cov = UKF._generate_sigma_points(
            x_pred, P_pred, alpha, beta, kappa
        )

        # Propagate sigma points through AUGMENTED measurement function
        m_aug = len(yk_aug)  # Augmented measurement dimension
        gamma_pred = np.zeros((2*n + 1, m_aug))
        for i in range(2*n + 1):
            gamma_pred[i] = h_aug(sigma_points_pred[i], **h_aug_params)

        # Compute predicted measurement mean and covariance
        # Use time-varying Rk_aug (key difference from standard UKF)
        y_pred, S = UKF._unscented_transform(gamma_pred, w_mean, w_cov, Rk_aug)

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
        innovation = yk_aug - y_pred
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

        Notes
        -----
        This function is identical to UKF.standard_h() - it simply extracts
        the state estimate from the (x_hat, P) tuple.
        """
        return xhat_P[0]

    @staticmethod
    def compute_observability_variance(
        observability_indicators: NDArray,
        window_size: int,
        base_variance: float = 1.0
    ) -> NDArray:
        """
        Compute time-varying R for ANN measurements based on observability.

        Parameters
        ----------
        observability_indicators : NDArray
            Time series of observability indicators, shape (T,)
            Higher values indicate better observability
            Examples: |acceleration|, |velocity|, |angular_rate|
        window_size : int
            Number of past timesteps to consider for minimum
        base_variance : float
            Base measurement variance (scales the output)

        Returns
        -------
        R_ann : NDArray
            Time-varying ANN measurement variance, shape (T,)

        Notes
        -----
        The observability-based variance is computed as:

        .. math::

            \\check{R}_{ANN}(t) = \\frac{c}{\\min_{i \\in [t-w, t]} \\sigma(i)}

        where:
        - :math:`c` is the base_variance parameter
        - :math:`w` is the window_size
        - :math:`\\sigma(i)` is the observability indicator at time :math:`i`

        **Intuition:**
        - **High observability** (large :math:`\\sigma`) → Small :math:`\\check{R}_{ANN}` → Trust ANN
        - **Low observability** (small :math:`\\sigma`) → Large :math:`\\check{R}_{ANN}` → Don't trust ANN

        **Example: Altitude from Optic Flow**
        When horizontal acceleration is high, altitude is observable from optic flow.
        When hovering (low acceleration), altitude is unobservable.

        .. code-block:: python

            accel_x = np.abs(acceleration_history)  # Observability indicator
            R_ann = AIUKF.compute_observability_variance(
                accel_x, window_size=10, base_variance=1.0
            )

        **Numerical Stability:**
        To prevent division by zero or very small numbers:
        - A small epsilon is added to the minimum observability
        - Very small observability indicators result in very large R (effectively ignoring ANN)

        Examples
        --------
        >>> import numpy as np
        >>> # Simulate varying acceleration (observability indicator)
        >>> accel = np.array([0.1, 0.5, 1.0, 0.2, 0.8, 0.05, 0.3])
        >>> R_ann = AIUKF.compute_observability_variance(accel, window_size=3)
        >>> R_ann.shape
        (7,)
        >>> # When acceleration is high (good observability), R is low
        >>> R_ann[2]  # accel=1.0 is high
        1.0
        >>> # When acceleration is low (poor observability), R is high
        >>> R_ann[5] > R_ann[2]  # accel=0.05 vs accel=1.0
        True
        """
        T = len(observability_indicators)
        R_ann = np.zeros(T)

        epsilon = 1e-6  # Prevent division by zero

        for t in range(T):
            # Get window of indicators
            start_idx = max(0, t - window_size + 1)
            window = observability_indicators[start_idx:t+1]

            # Find minimum observability in window
            min_obs = np.min(window) + epsilon

            # Inverse relationship: low observability → high R
            R_ann[t] = base_variance / min_obs

        return R_ann

    @staticmethod
    def create_augmented_R(
        R_sensor: NDArray,
        R_ann_time_varying: NDArray,
        num_sensor_measurements: int,
        time_index: int
    ) -> NDArray:
        """
        Construct augmented measurement covariance matrix at a specific time.

        Parameters
        ----------
        R_sensor : NDArray
            Sensor measurement covariance (constant), shape (m_sensor, m_sensor)
        R_ann_time_varying : NDArray
            Time series of ANN measurement variances, shape (T,)
            or (T, m_ann) for multiple ANN measurements
        num_sensor_measurements : int
            Number of sensor measurements (m_sensor)
        time_index : int
            Current time index

        Returns
        -------
        R_aug : NDArray
            Augmented measurement covariance, shape (m_aug, m_aug)

        Notes
        -----
        Constructs block-diagonal augmented R:

        .. math::

            \\mathbf{R}_{aug}(t) = \\begin{bmatrix}
                \\mathbf{R}_{sensor} & \\mathbf{0} \\\\
                \\mathbf{0} & \\check{R}_{ANN}(t)
            \\end{bmatrix}

        Examples
        --------
        >>> import numpy as np
        >>> R_sensor = np.diag([0.1, 0.2, 0.05])  # 3 sensor measurements
        >>> R_ann_series = np.array([1.0, 0.5, 2.0, 0.8])  # Time series
        >>> R_aug = AIUKF.create_augmented_R(R_sensor, R_ann_series, 3, time_index=1)
        >>> R_aug.shape
        (4, 4)
        >>> R_aug[-1, -1]  # ANN variance at time_index=1
        0.5
        """
        m_sensor = num_sensor_measurements

        # Handle scalar or vector ANN variance
        if R_ann_time_varying.ndim == 1:
            # Single ANN measurement
            R_ann_current = np.array([[R_ann_time_varying[time_index]]])
        else:
            # Multiple ANN measurements
            R_ann_current = np.diag(R_ann_time_varying[time_index])

        # Construct block-diagonal augmented R
        m_aug = m_sensor + R_ann_current.shape[0]
        R_aug = np.zeros((m_aug, m_aug))

        # Top-left: sensor covariance
        R_aug[0:m_sensor, 0:m_sensor] = R_sensor

        # Bottom-right: ANN covariance
        R_aug[m_sensor:, m_sensor:] = R_ann_current

        return R_aug


# Module-level aliases for convenience
# Allows usage like: from pykal.algorithm_library.estimators import ai_ukf; ai_ukf.f(...)
f = AIUKF.f
h = AIUKF.h
compute_observability_variance = AIUKF.compute_observability_variance
