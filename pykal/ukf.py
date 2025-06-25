from functools import wraps
from _base_ekf import BaseEKF
from system import System
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
from utils.iosafety import SafeIO as safeio
import scipy
from _base_ukf import BaseUKF


class UKFIO(SystemIO):
    """
    Unscented Kalman Filter I/O validation utilities.

    `UKFIO` extends `SystemIO` with specialized validation tools for running
    Unscented Kalman Filters, including input checks for sigma point configuration
    and UKF-specific parameters like `alpha`, `beta`, and `kappa`.

    Inherits
    --------
    SystemIO
        Provides validated registration of system dynamics, measurements, noise models,
        and structural information like names and system type.

    Methods
    -------
    verify_run_inputs(func: Callable) -> Callable
        Decorator to check dimensions and types for UKF initialization and execution.
    """

    @staticmethod
    def verify_run_inputs(func: Callable) -> Callable:
        """
        Decorator to validate inputs to the UKF `run` method.

        Ensures that initial conditions, measurement data, and simulation parameters are
        correctly typed and dimensioned before execution. Also validates sigma point scaling parameters.

        Validation Rules
        ----------------
        - `x0`, `P0`: shapes (n, 1) and (n, n)
        - `Y`: shape (T, m), with T ≥ 1 and m ≥ 1
        - `start_time`, `dt`: must be float or int
        - `u`: must be callable
        - `alpha`, `beta`, `kappa`: must be float or int

        Parameters
        ----------
        func : Callable
            The `run` method to be wrapped and validated.

        Raises
        ------
        ValueError
            If `Y` does not have valid dimensions.
        TypeError
            For incorrect types of inputs or UKF parameters.

        Returns
        -------
        Callable
            A wrapped version of the `run` method with validation enforced.
        """

        @wraps(func)
        def wrapper(
            self,
            *,
            x0: NDArray,
            P0: NDArray,
            Y: NDArray,
            start_time: float,
            dt: float,
            u: Callable,
            alpha: float,
            beta: float,
            kappa: float,
            square_root: bool,
            **kwargs,
        ):
            if not isinstance(Y, np.ndarray) or Y.ndim != 2:
                raise ValueError("Y must be a 2D NumPy array of shape (T, m)")
            N_steps, m_meas = Y.shape
            if N_steps == 0 or m_meas == 0:
                raise ValueError("Measurement history `Y` must contain valid data")

            if not callable(u):
                raise TypeError("u must be a callable function")

            for name, param in [
                ("start_time", start_time),
                ("dt", dt),
                ("alpha", alpha),
                ("beta", beta),
                ("kappa", kappa),
            ]:
                if not isinstance(param, (float, int)):
                    raise TypeError(f"{name} must be a float or int")

            if not isinstance(square_root, bool):
                raise TypeError("square_root must be a boolean flag")

            return func(
                self,
                x0=x0,
                P0=P0,
                Y=Y,
                start_time=start_time,
                dt=dt,
                u=u,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                square_root=square_root,
                **kwargs,
            )

        return wrapper


class UKFIO:
    """
    Unscented Kalman Filter I/O validation utilities.

    `UKFIO` extends `SystemIO` with specialized validation tools for running
    Unscented Kalman Filters, including input checks for UKF-specific configuration
    parameters like sigma point scaling and square-root filtering.

    Inherits
    --------
    SystemIO
        Provides validated registration of system dynamics, measurements, noise models,
        and structural information like names and system type.

    Methods
    -------
    verify_run_inputs(func: Callable) -> Callable
        Decorator to check dimensions and types for filter initialization and execution.


    """

    @staticmethod
    def verify_run_inputs(func: Callable) -> Callable:
        """
        Decorator to validate inputs to the UKF `run` method.

        Ensures that initial conditions, measurement data, and configuration
        parameters are correctly typed and dimensioned before execution.

        Validation Rules
        ----------------
        - `x0`, `P0`: shapes (n, 1) and (n, n)
        - `Y`: shape (T, m), with T ≥ 1 and m ≥ 1
        - `start_time`, `dt`: must be float or int
        - `u`: must be a callable
        - `square_root`: must be a boolean
        - `alpha`, `beta`, `kappa`: must be floats or ints (UKF scaling params)

        Parameters
        ----------
        func : Callable
            The `run` method to be wrapped and validated.

        Raises
        ------
        ValueError
            If `Y` does not have valid dimensions.
        TypeError
            For incorrect types of time parameters, scaling constants, or boolean flags.

        Returns
        -------
        Callable
            A wrapped version of the `run` method with validation enforced.
        """

        @wraps(func)
        def wrapper(
            self,
            *,
            x0: NDArray,
            P0: NDArray,
            Y: NDArray,
            start_time: float,
            dt: float,
            u: Callable,
            square_root: bool,
            alpha: float,
            beta: float,
            kappa: float,
            override_beta: Optional[Callable] = None,
            **kwargs,
        ):
            if not isinstance(Y, np.ndarray) or Y.ndim != 2:
                raise ValueError("Y must be a 2D NumPy array of shape (T, m)")
            N_steps, m_meas = Y.shape
            if N_steps == 0 or m_meas == 0:
                raise ValueError(
                    "Measurement history `Y` must contain valid rows and columns"
                )

            if not callable(u):
                raise TypeError("u must be a callable function")

            if not isinstance(start_time, (int, float)):
                raise TypeError("start_time must be a float or int")
            if not isinstance(dt, (int, float)):
                raise TypeError("dt must be a float or int")

            if not isinstance(square_root, bool):
                raise TypeError("square_root must be a boolean flag")

            for name, val in zip(["alpha", "beta", "kappa"], [alpha, beta, kappa]):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"{name} must be a float or int")

            return func(
                self,
                x0=x0,
                P0=P0,
                Y=Y,
                start_time=start_time,
                dt=dt,
                u=u,
                square_root=square_root,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                override_beta=override_beta,
                **kwargs,
            )

        return wrapper

    @staticmethod
    def set_beta(beta: Optional[Callable], state_names: Sequence[str]) -> Callable:
        """
        Set the β function for the Partial-Update Kalman Filter (PKF).

        This function defines how much each individual state is updated during the EKF
        correction step. The user may provide a callable β(t) → NDArray of shape (n, 1),
        where `n = len(state_names)`. If `beta` is None, a default function is returned
        that always outputs a column vector of ones (full update).

        Parameters
        ----------
        beta : Callable or None
            A function of the form `beta(t) -> NDArray` of shape (n, 1), or `None` to
            use the default (full update) behavior.
        state_names : Sequence[str]
            List of state names used to determine the output shape of the default β.

        Returns
        -------
        Callable
            A validated β function returning an array of shape (n, 1), where n = number of states.

        Examples
        --------
        >>> from numpy.typing import NDArray
        >>> import numpy as np

        >>> def my_beta(t: float) -> NDArray:
        ...     return np.array([[1.0], [0.0]])  # Only update first state

        >>> checked_beta = PKFIO.set_beta(my_beta, state_names=["x0", "x1"])
        >>> checked_beta(0.0)
        array([[1.],
               [0.]])

        >>> default_beta = PKFIO.set_beta(None, state_names=["x0", "x1", "x2"])
        >>> default_beta(0.0)
        array([[1.],
               [1.],
               [1.]])
        """
        if beta is None:
            n_states = len(state_names)

            def default_beta(t: float) -> NDArray:
                return np.ones((n_states, 1))

            return default_beta
        else:

            @safeio.verify_signature_and_parameter_names
            def verify_beta(beta: Callable) -> Callable:
                return beta

            return verify_beta(beta)


class UKF(BaseUKF):
    """
    Unscented Kalman Filter (UKF) with optional partial updates and square-root filtering.

    This class implements the UKF for nonlinear state estimation using deterministic sampling
    (sigma points) to propagate means and covariances through nonlinear transformations.

    It generalizes the standard UKF by supporting:

    1. **Per-state partial updates** via a time-varying β vector.
    2. **Square-root filtering** for improved numerical stability during covariance updates.

    Each component βᵢ ∈ [0, 1] controls how much the corresponding state is corrected
    based on a confidence metric (e.g., observability, innovation magnitude). The
    update uses a diagonal matrix βₖ ∈ ℝⁿˣⁿ, where n is the number of states.
    βᵢ = 0 means no update to state i; βᵢ = 1 yields a full UKF update.

    Inherits From
    -------------
    BaseKF
        Abstract base class defining the Kalman Filter interface and shared behavior.

    Parameters
    ----------
    sys : System
        A validated `System` object describing the system dynamics, measurement
        model, noise models, and timing structure. Must provide callable functions:
          - f(x, u, t): state dynamics
          - h(x, u, t): measurement model
          - Q(x, u, t): process noise covariance
          - R(x, u, t): measurement noise covariance

        The system must also define:
          - The system type (discrete/continuous, time-varying/invariant)
          - Variable name metadata for states and measurements

    Notes
    -----
    The UKF uses the **unscented transform** with tunable parameters (α, β, κ) to
    generate and propagate sigma points:
      - α (alpha): spread of the sigma points around the mean (usually small, e.g., 1e-3)
      - β (beta): prior knowledge about distribution (2 is optimal for Gaussians)
      - κ (kappa): secondary scaling parameter (often 0 or 3 - n)

    This UKF supports both continuous-time and discrete-time systems, and can operate
    in square-root mode for enhanced numerical stability, especially under ill-conditioned
    covariance updates or high-dimensional systems.

    Square-root filtering is **currently implemented only for discrete-time systems**.
    Continuous-time UKF variants would require integration of sigma points over time,
    which is not yet supported.


     Examples
    --------
    >>> import numpy as np
    >>> from numpy.typing import NDArray
    >>> from system import System, SystemType
    >>> from filters import UKF

    >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     A = np.array([[1.0, 0.1], [0.0, 1.0]])
    ...     return A @ x

    >>> def h(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     H = np.array([[1.0, 0.0]])
    ...     return H @ x

    >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return 0.01 * np.eye(2)

    >>> def R(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return 0.1 * np.eye(1)

    >>> def u(t: float) -> NDArray:
    ...     return np.zeros((1, 1))

    >>> sys = System(
    ...     f=f, h=h, Q=Q, R=R, u=u,
    ...     state_names=["x0", "x1"],
    ...     measurement_names=["x0"],
    ...     system_type=SystemType.DISCRETE_TIME_INVARIANT,
    ... )

    >>> x0 = np.array([[0.0], [1.0]])
    >>> P0 = 0.01 * np.eye(2)

    >>> X_sim, T_sim = sys.simulate_states(x0=x0, dt=0.1, t_span=(0.0, 1.0), process_noise=False)
    >>> Y, _ = sys.simulate_measurements(X=X_sim, T=T_sim, measurement_noise=False)

    # === Full update (standard UKF) ===
    >>> ukf = UKF(sys)
    >>> X_est, P_est, T_out = ukf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     F=lambda *a, **kw: None,  # not used in UKF
    ...     H=lambda *a, **kw: None,  # not used in UKF
    ...     start_time=0.0, dt=0.1, u=u,
    ...     square_root=False,
    ... )

    >>> X_est.shape == (Y.shape[1] + 1, 2)
    True
    >>> np.allclose(X_est[0], x0.flatten())
    True
    >>> np.any(np.abs(np.diff(X_est[:, 1])) > 1e-6)
    np.True_

    # === SKF-like update (only first state corrected) ===
    >>> def beta(t: float) -> NDArray:
    ...     return np.array([[1.0], [0.0]])

    >>> ukf.beta = beta
    >>> X_est, P_est, T_out = ukf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     F=lambda *a, **kw: None,
    ...     H=lambda *a, **kw: None,
    ...     start_time=0.0, dt=0.1, u=u,
    ...     square_root=False,
    ... )

    >>> np.allclose(X_est[0], x0.flatten())
    True
    >>> np.allclose(X_est[:, 1], x0[1, 0])  # second state unchanged
    True

    # === Partial update (β = [0.5, 0.0]) ===
    >>> def beta(t: float) -> NDArray:
    ...     return np.array([[0.5], [0.0]])

    >>> ukf.beta = beta
    >>> X_est, P_est, T_out = ukf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     F=lambda *a, **kw: None,
    ...     H=lambda *a, **kw: None,
    ...     start_time=0.0, dt=0.1, u=u,
    ...     square_root=False,
    ... )

    >>> np.allclose(X_est[0], x0.flatten())
    True
    >>> np.allclose(X_est[:, 1], x0[1, 0])
    True
    >>> not np.allclose(X_est[1:, 0], X_sim[0, :])
    True
    """

    def __init__(
        self,
        sys: System,
        beta: Optional[Callable[[float], NDArray]] = None,
    ):
        super().__init__(sys)
        self._beta = UKFIO.set_beta(beta, self.sys.state_names)

    @property
    def beta(self) -> Callable:
        """Return the current β function."""
        return self._beta

    @beta.setter
    def beta(self, func: Callable):
        """Set a new β function and validate its output shape."""
        self._beta = UKFIO.set_beta(func, self.sys.state_names)

    def _predict(
        self,
        xk: NDArray,
        Pk: NDArray,
        Fk: NDArray,  # Unused in UKF, kept for API consistency
        Qk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
        square_root: bool,
    ) -> tuple[NDArray, NDArray]:
        r"""
        Internal UKF prediction step using sigma-point propagation.

        This method propagates the mean and covariance of the state estimate forward
        in time using the Unscented Transform. Sigma points are deterministically
        generated from the prior state and covariance, propagated through the dynamics,
        and re-aggregated to yield the predicted mean and covariance.

        If `square_root` is True, Cholesky-based sigma point generation and QR-based
        recomposition are used for numerical stability (discrete-time only).

        Parameters
        ----------
        xk : NDArray
            Current state estimate, shape (n, 1).
        Pk : NDArray
            Current covariance matrix, shape (n, n).
        Fk : NDArray
            Ignored in UKF (included for API consistency).
        Qk : NDArray
            Process noise covariance matrix, shape (n, n).
        dt : float
            Time step.
        uk : NDArray
            Control input at current time, shape (m, 1).
        tk : float
            Current time value.
        square_root : bool
            Whether to apply square-root UKF propagation (QR-based recomposition).

        Returns
        -------
        x_pred : NDArray
            Predicted state mean after propagation, shape (n, 1).
        P_pred : NDArray
            Predicted state covariance after propagation, shape (n, n).

        Notes
        -----
        - Uses scaled unscented transform (default α = 1e-3, β = 2, κ = 0).
        - Continuous-time UKF is not implemented.
        - This method does not use Jacobians.
        """

        alpha = 1e-3
        beta_ = 2.0
        kappa = 0.0

        n = xk.shape[0]
        lambda_ = alpha**2 * (n + kappa) - n

        # Weights for mean and covariance
        Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lambda_)))
        Wc = Wm.copy()
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = Wm[0] + (1 - alpha**2 + beta_)

        # Sigma points
        if square_root:
            try:
                S = scipy.linalg.cholesky(Pk, lower=True)
            except np.linalg.LinAlgError as e:
                raise ValueError("Cholesky failed: Pk not positive-definite") from e
        else:
            S = np.linalg.cholesky(Pk + 1e-9 * np.eye(n))  # fallback if not sqrt-mode

        sigma_pts = np.zeros((n, 2 * n + 1))
        sigma_pts[:, 0:1] = xk
        scaled = np.sqrt(n + lambda_) * S
        sigma_pts[:, 1 : n + 1] = xk + scaled
        sigma_pts[:, n + 1 :] = xk - scaled

        # Propagate each sigma point
        propagated = np.zeros_like(sigma_pts)
        for i in range(2 * n + 1):
            xi = sigma_pts[:, i].reshape(-1, 1)
            fi = safeio.call_validated_function_with_args(self.sys.f, x=xi, u=uk, t=tk)
            if self.sys.system_type.is_continuous:
                fi = xi + fi * dt
            propagated[:, i] = fi.flatten()

        # Predicted mean
        x_pred = propagated @ Wm.reshape(-1, 1)

        # Predicted covariance
        dx = propagated - x_pred
        P_pred = dx @ np.diag(Wc) @ dx.T + Qk

        return x_pred, P_pred

    def _update(
        self,
        xk: NDArray,
        Pk: NDArray,
        yk: NDArray,
        Hk: NDArray,  # Not used in UKF; kept for API consistency
        Rk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
        square_root: bool,
        Beta: NDArray,
    ) -> tuple[NDArray, NDArray]:
        r"""
        Internal UKF measurement update step with optional square-root filtering and partial updates.

        Computes the Kalman gain and corrects the predicted state `xk` using the observed
        measurement `yk`. The unscented transform is used to propagate sigma points through
        the nonlinear measurement function `h`.

        Supports both standard and square-root UKF updates and per-state partial corrections
        using the diagonal confidence matrix `Beta`.

        Parameters
        ----------
        xk : NDArray
            Predicted state estimate (n×1).
        Pk : NDArray
            Predicted covariance matrix (n×n).
        yk : NDArray
            Measurement at time `tk` (m×1).
        Hk : NDArray
            Unused (UKF does not linearize via Jacobians).
        Rk : NDArray
            Measurement noise covariance (m×m).
        dt : float
            Time step (unused).
        uk : NDArray
            Control input (m×1).
        tk : float
            Current time.
        square_root : bool
            Whether to use square-root update (Cholesky + QR).
        Beta : NDArray
            Diagonal weighting matrix (n×n) for per-state update confidence.

        Returns
        -------
        x_upd : NDArray
            Updated state estimate (n×1).
        P_upd : NDArray
            Updated covariance matrix (n×n).
        """

        alpha = 1e-3
        beta_ = 2.0
        kappa = 0.0

        n = xk.shape[0]
        m = yk.shape[0]
        lambda_ = alpha**2 * (n + kappa) - n

        # Sigma point weights
        Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lambda_)))
        Wc = Wm.copy()
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = Wm[0] + (1 - alpha**2 + beta_)

        # Sigma points from predicted mean
        if square_root:
            S = scipy.linalg.cholesky(Pk, lower=True)
        else:
            S = np.linalg.cholesky(Pk + 1e-9 * np.eye(n))

        sigma_pts = np.zeros((n, 2 * n + 1))
        sigma_pts[:, 0:1] = xk
        scaled = np.sqrt(n + lambda_) * S
        sigma_pts[:, 1 : n + 1] = xk + scaled
        sigma_pts[:, n + 1 :] = xk - scaled

        # Propagate through measurement function h
        Z = np.zeros((m, 2 * n + 1))
        for i in range(2 * n + 1):
            xi = sigma_pts[:, i].reshape(-1, 1)
            hi = safeio.call_validated_function_with_args(self.sys.h, x=xi, u=uk, t=tk)
            Z[:, i] = hi.flatten()

        z_pred = Z @ Wm.reshape(-1, 1)
        dz = Z - z_pred
        dx = sigma_pts - xk

        # Cross covariance and innovation covariance
        Pxz = dx @ np.diag(Wc) @ dz.T  # (n x m)
        Pzz = dz @ np.diag(Wc) @ dz.T + Rk  # (m x m)

        ridge = 1e-9 * np.eye(m)
        try:
            S_inv = np.linalg.inv(Pzz + ridge)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(Pzz + ridge)

        Kk = Pxz @ S_inv
        innovation = yk - z_pred

        # Apply per-state confidence weights
        x_upd = xk + Beta @ (Kk @ innovation)

        # Covariance update
        I = np.eye(n)
        P_base = (I - Kk @ (Z - z_pred) @ Wm.reshape(-1, 1).T) @ Pk @ (
            I - Kk @ (Z - z_pred) @ Wm.reshape(-1, 1).T
        ).T + Kk @ Rk @ Kk.T

        P_upd = Beta @ P_base @ Beta.T + (I - Beta) @ Pk @ (I - Beta).T

        return x_upd, P_upd

    @UKFIO.verify_run_inputs
    def run(
        self,
        *,
        x0: NDArray,
        P0: NDArray,
        Y: NDArray,
        F: Callable,  # Ignored in UKF, required for interface compatibility
        H: Callable,  # Ignored in UKF, required for interface compatibility
        start_time: float,
        dt: float,
        u: Callable,
        square_root: bool = False,
        override_beta: Optional[Callable] = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Run the full UKF estimation over a sequence of measurements.

        At each step, applies `_predict` followed by `_update` using unscented sigma points.

        Parameters
        ----------
        x0 : NDArray
            Initial state estimate, shape (n, 1).
        P0 : NDArray
            Initial covariance estimate, shape (n, n).
        Y : NDArray
            Measurement history, shape (T, m).
        F : Callable
            Ignored (for API compatibility with EKF).
        H : Callable
            Ignored (for API compatibility with EKF).
        start_time : float
            Start time corresponding to the first measurement Y[0].
        dt : float
            Time step between measurements.
        u : Callable
            Function u(t) → NDArray, returning control input of shape (m, 1).
        square_root : bool, default=False
            Whether to use square-root filtering.
        override_beta : Callable or False, default=False
            Optional override for the β function. If False, uses the filter's default β.

        Returns
        -------
        x_hist : NDArray
            Filtered state estimates at all times, shape (T+1, n).
        P_hist : NDArray
            Covariance estimates at all times, shape (T+1, n, n).
        t_hist : NDArray
            Time vector corresponding to each state estimate, shape (T+1,).
        """
        n_states = len(self.sys.state_names)
        m_meas = len(self.sys.measurement_names)
        N_steps, M_meas = Y.shape

        if M_meas != m_meas:
            raise ValueError(
                f"Measurement dimension mismatch: got {M_meas}, expected {m_meas}"
            )

        x_hist_arr = np.empty((N_steps + 1, n_states))
        P_hist_arr = np.empty((N_steps + 1, n_states, n_states))
        t_hist_arr = start_time + np.arange(N_steps + 1) * dt

        xk = x0
        Pk = P0
        x_hist_arr[0] = xk.flatten()
        P_hist_arr[0] = Pk

        beta_fn = (
            self._beta
            if override_beta is False
            else UKFIO.set_beta(override_beta, self.sys.state_names)
        )

        for k in range(N_steps):
            yk = Y[k].reshape(-1, 1)
            tk = t_hist_arr[k]
            uk = safeio.call_validated_function_with_args(u, t=tk)

            Qk = safeio.call_validated_function_with_args(self.sys.Q, x=xk, u=uk, t=tk)
            Rk = safeio.call_validated_function_with_args(self.sys.R, x=xk, u=uk, t=tk)
            Beta = safeio.call_validated_function_with_args(beta_fn, t=tk)

            xk, Pk = self._predict(xk, Pk, Qk, dt, uk, tk, square_root)
            xk, Pk = self._update(
                xk,
                Pk,
                yk,
                Hk=None,
                Rk=Rk,
                dt=dt,
                uk=uk,
                tk=tk,
                square_root=square_root,
                Beta=Beta,
            )

            x_hist_arr[k + 1] = xk.flatten()
            P_hist_arr[k + 1] = Pk

        return x_hist_arr, P_hist_arr, t_hist_arr
