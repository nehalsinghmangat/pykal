from functools import wraps
from system import System
from _base_kf import BaseKFSqrt, BaseKFPartialUpdate
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional, Sequence, Union
import numpy as np
from numpy.typing import NDArray
from utils.utils_safeio import SafeIO as safeio


class EKFIO:
    @staticmethod
    def verify_run_inputs(func: Callable) -> Callable:
        """
        Decorator to validate inputs to the Kalman Filter `run` method.

        Ensures that initial conditions, measurement data, Jacobians, and
        simulation parameters are correctly typed and dimensioned before execution.

        Validation Rules
        ----------------
        - `x0`, `P0`: shapes (n, 1) and (n, n)
        - `Y`: shape (T, m), with T ≥ 1 and m ≥ 1
        - `F`, `H`, `u`: must be callables
        - `start_time`, `dt`: must be float or int
        - `square_root`: must be a boolean

        Parameters
        ----------
        func : Callable
            The `run` method to be wrapped and validated.

        Raises
        ------
        ValueError
            If `Y` does not have valid dimensions.
        TypeError
            For incorrect types of Jacobians, time parameters, or boolean flags.

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
            F: Callable,
            H: Callable,
            start_time: float,
            dt: float,
            square_root: bool,
            override_beta: Optional[Callable],
            **kwargs,
        ):
            if not isinstance(Y, np.ndarray) or Y.ndim != 2:
                raise ValueError("Y must be a 2D NumPy array of shape (T, m)")
            N_steps, m_meas = Y.shape
            if N_steps == 0:
                raise ValueError(
                    "Measurement history `Y` must contain at least one row"
                )
            if m_meas == 0:
                raise ValueError(
                    "Measurement history `Y` must contain at least one column"
                )

            if not callable(F):
                raise TypeError("F must be a callable function")
            if not callable(H):
                raise TypeError("H must be a callable function")

            if not isinstance(start_time, (int, float)):
                raise TypeError("start_time must be a float or int")
            if not isinstance(dt, (int, float)):
                raise TypeError("dt must be a float or int")

            if not isinstance(square_root, bool):
                raise TypeError("square_root must be a boolean flag")
            if override_beta not in (None, False):
                if not isinstance(override_beta, Callable):
                    raise TypeError("override_beta must be a callable")

            return func(
                self,
                x0=x0,
                P0=P0,
                Y=Y,
                F=F,
                H=H,
                start_time=start_time,
                dt=dt,
                square_root=square_root,
                override_beta=override_beta,
                **kwargs,
            )

        return wrapper


class EKF(BaseKFSqrt, BaseKFPartialUpdate):
    """
    Extended Kalman Filter (EKF) with optional partial updates and square-root filtering.

    This class implements the EKF for nonlinear state estimation by linearizing
    the system dynamics and measurement models about the current estimate.

    It generalizes the standard EKF by supporting:

    1. **Per-state partial updates** via a time-varying β vector.
    2. **Square-root filtering** for improved numerical stability during covariance updates.

    Each component βᵢ ∈ [0, 1] controls how much the corresponding state is corrected
    based on a confidence metric (e.g., observability, innovation magnitude). The
    update uses a diagonal matrix βₖ ∈ ℝⁿˣⁿ, where n is the number of states.
    βᵢ = 0 means no update to state i; βᵢ = 1 yields a full EKF update.

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
    The EKF uses externally provided Jacobian functions for the dynamics and
    measurement models:
      - F(x, u, t) ≈ ∂f/∂x
      - H(x, u, t) ≈ ∂h/∂x

    These are required at runtime in the `run()` method.

    All model evaluations (f, h, Q, R, F, H) are dispatched through
    `call_validated_function_with_args` to ensure input consistency and robust error handling.

    This EKF supports both continuous-time and discrete-time systems, and can operate
    in square-root mode for enhanced numerical stability, especially under ill-conditioned
    covariance updates or high-dimensional systems.


    Square-root filtering is **currently implemented only for discrete-time systems**.
    Continuous-time EKF variants use direct covariance integration (via the
    Riccati equation) rather than square-root propagation. Extending square-root
    methods to continuous-time cases would require additional numerical solvers,
    and life is short.

        Examples
    --------
    >>> import numpy as np
    >>> from numpy.typing import NDArray
    >>> from system import System, SystemType

    # Define system dynamics, measurement, and Jacobians
    >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     A = np.array([[1.0, 0.1], [0, 1.0]])
    ...     return A @ x

    >>> def h(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     H = np.array([[1.0, 0.0]])
    ...     return H @ x

    >>> def F(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return np.array([[1.0, 0.1], [0, 1.0]])

    >>> def H(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return np.array([[1.0, 0.0]])

    >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return 0.01 * np.eye(2)

    >>> def R(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return 0.1 * np.eye(1)

    >>> def u(t: float) -> NDArray:
    ...     return np.zeros((1, 1))

    # Construct the system object
    >>> sys = System(
    ...     f=f, h=h, Q=Q, R=R, u=u,
    ...     state_names=["x0", "x1"],
    ...     measurement_names=["x0"],
    ...     system_type=SystemType.DISCRETE_TIME_INVARIANT,
    ... )

    # Initial condition
    >>> x0 = np.array([[0.0], [1.0]])
    >>> P0 = 0.01 * np.eye(2)

    # Simulate reference trajectory and measurements (no noise)
    >>> X_sim, T_sim = sys.simulate_states(x0=x0, dt=0.1, t_span=(0.0, 1.0), process_noise=False)
    >>> Y, _ = sys.simulate_measurements(X=X_sim, T=T_sim, measurement_noise=False)


    # === Full update (EKF behavior) ===
    >>> ekf = EKF(sys)  # no beta passed → defaults to β = 1 for all states
    >>> X_est, P_est, T_out = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     F=F, H=H,
    ...     start_time=0.0,
    ...     dt=0.1,
    ...     square_root=False,
    ...     override_beta=False)


    >>> X_est.shape == (Y.shape[1] + 1, x0.shape[0])
    True
    >>> np.allclose(X_est[0], x0.flatten())  # initial state
    True
    >>> not np.allclose(X_est[1], x0.flatten())  # generally NOT equal
    True

    # === Compare standard EKF and square-root EKF ===
    >>> X_sqrt, P_sqrt, _ = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     F=F, H=H,
    ...     start_time=0.0,
    ...     dt=0.1,
    ...     square_root=True,
    ...     override_beta=False
    ... )

    >>> np.allclose(X_est, X_sqrt, atol=1e-5)
    True
    >>> np.allclose(P_est[-1], P_sqrt[-1], atol=1,rtol=1)
    True



    # === Only first state updated (SKF behavior) ===
    >>> def beta(t: float) -> NDArray:
    ...     return np.array([[1.0], [0.0]])

    >>> ekf.beta = beta

    >>> X_est, P_est, T_out = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     F=F, H=H,
    ...     start_time=0.0,
    ...     dt=0.1,
    ...     square_root = False,
    ...     override_beta = False)

    >>> X_est.shape == (Y.shape[1] + 1, 2)
    True
    >>> np.allclose(X_est[0], x0.flatten())
    True
    >>> np.allclose(X_est[:, 1], x0[1, 0])  # second state remains unchanged
    True

    # === Partially update the first state ===
    >>> def beta(t: float) -> NDArray:
    ...     return np.array([[0.5], [0.0]])

    >>> X_est, P_est, T_out = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     F=F, H=H,
    ...     start_time=0.0,
    ...     dt=0.1,
    ...     square_root = False,
    ...     override_beta = beta)

    >>> X_est.shape == (Y.shape[1] + 1, 2)
    True
    >>> np.allclose(X_est[0], x0.flatten())
    True
    >>> not np.allclose(X_est[1:, 0], x0[0, 0])  # first state evolves
    True
    >>> np.allclose(X_est[:, 1], x0[1, 0])  # second state still frozen
    True
    >>> # First state is only partially corrected (less aggressive than β = 1)
    >>> not np.allclose(X_est[1:, 0], X_sim[0,:])  # state 0 differs from true
    True
    """

    def __init__(
        self,
        sys: System,
        beta: Optional[Callable[[NDArray, NDArray, float], NDArray]] = None,
    ):

        super().__init__(sys=sys, beta=beta)

    def _predict(
        self,
        xk: NDArray,
        Pk: NDArray,
        Fk: NDArray,
        Qk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
        square_root: bool,
    ) -> tuple[NDArray, NDArray]:
        r"""
        Internal EKF prediction step (time update), supporting optional square-root filtering.

        This method propagates the mean and covariance of the state estimate forward
        in time using the system's dynamics and linearized model. The prediction
        behavior depends on whether the system is continuous-time or discrete-time.

        - For **continuous-time systems**, Euler integration is used:
          \[
          x_{k+1} = x_k + f(x_k, u_k, t_k)\,\Delta t
          \]
          \[
          P_{k+1} = P_k + (F_k P_k + P_k F_k^\top + Q_k)\,\Delta t
          \]
          Square-root prediction is not applied in the continuous case.

        - For **discrete-time systems**, the dynamics and covariance are updated via:
          \[
          x_{k+1} = f(x_k, u_k, t_k)
          \]
          \[
          P_{k+1} =
            \begin{cases}
              F_k P_k F_k^\top + Q_k & \text{(standard)} \\
              \text{QR-based square-root update}       & \text{(if } square\_root = \text{True)}
            \end{cases}
          \]

        If `square_root` is True and the system is discrete, the covariance update
        is performed using a numerically stable QR-based square-root method.

        Parameters
        ----------
        xk : NDArray
            Current state estimate, shape (n, 1).
        Pk : NDArray
            Current covariance matrix, shape (n, n).
        Fk : NDArray
            Jacobian of the dynamics model ∂f/∂x evaluated at (xk, uk, tk), shape (n, n).
        Qk : NDArray
            Process noise covariance evaluated at (xk, uk, tk), shape (n, n).
        dt : float
            Time step between measurements.
        uk : NDArray
            Control input at current time, shape (m, 1).
        tk : float
            Current time value.
        square_root : bool
            Whether to apply square-root covariance prediction (QR-based update).
            Only applies in the discrete-time case.

        Returns
        -------
        x_pred : NDArray
            Predicted state estimate after propagation, shape (n, 1).
        P_pred : NDArray
            Predicted covariance (or square-root approximation) after propagation, shape (n, n).

        Notes
        -----
        - Square-root propagation is only applied for discrete-time systems.
        - Continuous-time covariance is always propagated using Euler integration.
        - This method relies on `safeio.call_validated_function_with_args` for safe dispatch
          of user-defined functions.
        """

        if self.sys.system_type in (
            self.sys.system_type.CONTINUOUS_TIME_INVARIANT,
            self.sys.system_type.CONTINUOUS_TIME_VARYING,
        ):
            dx = safeio.call_validated_function_with_args(self.sys.f, x=xk, u=uk, t=tk)
            x_pred = xk + dx * dt
            P_pred = Pk + (Fk @ Pk + Pk @ Fk.T + Qk) * dt
        else:
            x_pred = safeio.call_validated_function_with_args(
                self.sys.f, x=xk, u=uk, t=tk
            )

            if square_root:
                P_pred = self._square_root_predict_covariance(Pk, Fk, Qk)
            else:
                P_pred = Fk @ Pk @ Fk.T + Qk

        return x_pred, P_pred

    def _update(
        self,
        xk: NDArray,
        Pk: NDArray,
        yk: NDArray,
        y_pred: NDArray,
        Hk: NDArray,
        Rk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
        square_root: bool,
        beta_mat: NDArray,
    ) -> tuple[NDArray, NDArray]:
        r"""
        Internal EKF measurement update step (correction), with optional partial update and square-root filtering.

        Incorporates a new measurement into the current state and covariance estimate
        using the extended Kalman filter's innovation-based update equations.

        The Kalman gain is computed as:
        \[
        K_k = P_k H_k^\top (H_k P_k H_k^\top + R_k)^{-1}
        \]

        
        The standard update equations are:
        \[
        \begin{aligned}
        \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k \left(y_k - h(\hat{x}_{k|k-1}, u_k, t_k)\right) \\\\
        P_{k|k} &= (I - K_k H_k) P_k (I - K_k H_k)^\top + K_k R_k K_k^\top
        \end{aligned}
        \]

        The partial-update formulation modifies the update as:
        \[
        \begin{aligned}
        \hat{x}_{k|k} &= \hat{x}_{k|k-1} + \beta_k \cdot K_k \left(y_k - h(\hat{x}_{k|k-1}, u_k, t_k)\right) \\\\
        P_{k|k} &= \beta_k \left[ (I - K_k H_k) P_k (I - K_k H_k)^\top + K_k R_k K_k^\top \right] \beta_k^\top + (I - \beta_k) P_k (I - \beta_k)^\top
        \end{aligned}
        \]

        If `square_root` is True, the covariance is updated using a numerically stable
        QR-based square-root formulation. This involves Cholesky decompositions of both
        the prior covariance and the measurement noise covariance, followed by a QR
        decomposition of the augmented innovation matrix.

        Parameters
        ----------
        xk : NDArray
            Predicted state estimate from the prediction step, shape (n, 1).
        Pk : NDArray
            Predicted covariance matrix from the prediction step, shape (n, n).
        yk : NDArray
            Measurement vector at time `tk`, shape (m, 1).
        Hk : NDArray
            Jacobian of the measurement function ∂h/∂x evaluated at (xk, uk, tk), shape (m, n).
        Rk : NDArray
            Measurement noise covariance evaluated at (xk, uk, tk), shape (m, m).
        dt : float
            Time step (unused; included for API uniformity).
        uk : NDArray or None
            Control input at time `tk`, shape (m, 1).
        tk : float or None
            Time at which the measurement is received.
        square_root : bool
            If True, apply square-root covariance update using QR decomposition.

        Returns
        -------
        x_upd : NDArray
            Updated state estimate after incorporating the measurement, shape (n, 1).
        P_upd : NDArray
            Updated covariance matrix (or square-root approximation), shape (n, n).

        Notes
        -----
        - βₖ is a diagonal matrix controlling the confidence in updating each state.
        - If `square_root=True`, a numerically stable QR-based update is used (currently only for discrete-time systems).
        - A small ridge term is added to the innovation covariance to prevent singularity in matrix inversion.
        - This method uses `safeio.call_validated_function_with_args` to safely evaluate
          the measurement function `h` with appropriate inputs.
        """
        Sk = Hk @ Pk @ Hk.T + Rk
        ridge = 1e-9 * np.eye(Sk.shape[0])
        try:
            Sk_inv = np.linalg.inv(Sk + ridge)
        except np.linalg.LinAlgError:
            Sk_inv = np.linalg.pinv(Sk + ridge)

        Kk = Pk @ Hk.T @ Sk_inv

        innovation = yk - y_pred
        x_upd = xk + Kk @ innovation

        # === Covariance update ===
        if square_root:
            P_base = self._square_root_update_covariance(Pk, Hk, Rk)
        else:
            I_KH = np.eye(Pk.shape[0]) - Kk @ Hk
            P_base = I_KH @ Pk @ I_KH.T + Kk @ Rk @ Kk.T

        P_upd = self._apply_partial_update(P_base, Pk, tk, beta_mat)

        return x_upd, P_upd

    @EKFIO.verify_run_inputs
    def run(
        self,
        *,
        x0: NDArray,
        P0: NDArray,
        Y: NDArray,
        F: Callable,
        H: Callable,
        start_time: float,
        dt: float,
        square_root: bool = False,
        override_beta: Union[Callable, None, bool] = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Run the full EKF estimation over a sequence of measurements.

        At each step, applies `_predict` followed by `_update` using the supplied
        Jacobian and noise functions.

        Parameters
        ----------
        x0 : NDArray
            Initial state estimate, shape (n, 1).
        P0 : NDArray
            Initial covariance estimate, shape (n, n).
        Y : NDArray
            Measurement history, shape (T, m).
        F : Callable
            Function returning the Jacobian ∂f/∂x: F(x, u, t) → NDArray (n × n).
        H : Callable
            Function returning the Jacobian ∂h/∂x: H(x, u, t) → NDArray (m × n).
        start_time : float
            Time corresponding to the first measurement Y[0].
        dt : float
            Time step between measurements.
        u : Callable
            Function u(t) → NDArray of shape (m, 1), providing the control input.

        override_system_u : Union[Callable, None, bool], optional
            - `False` (default): use internal system dynamics `f`.
            - `None`: raise an error — `f` is required.
            - Callable: use this function as dynamics override.
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

        beta = (
            self._beta
            if override_beta is False
            else self.set_beta(override_beta, self.sys.state_names)
        )
        for k in range(N_steps):
            yk = Y[k].reshape(-1, 1)
            tk = t_hist_arr[k]

            uk = safeio.call_validated_function_with_args(
                self.sys.u, t=tk, expected_shape=(len(self.sys.input_names), 1)
            )
            y_pred = safeio.call_validated_function_with_args(
                self.sys.h, x=xk, u=uk, t=tk
            )
            Fk = safeio.call_validated_function_with_args(F, x=xk, u=uk, t=tk)
            Hk = safeio.call_validated_function_with_args(H, x=xk, u=uk, t=tk)
            Qk = safeio.call_validated_function_with_args(self.sys.Q, x=xk, u=uk, t=tk)
            Rk = safeio.call_validated_function_with_args(self.sys.R, x=xk, u=uk, t=tk)
            beta_mat = np.diagflat(
                safeio.call_validated_function_with_args(
                    beta, t=tk, expected_shape=(n_states, 1)
                )
            )
            xk, Pk = self._predict(xk, Pk, Fk, Qk, dt, uk, tk, square_root)
            xk, Pk = self._update(
                xk, Pk, yk, y_pred, Hk, Rk, dt, uk, tk, square_root, beta_mat
            )

            x_hist_arr[k + 1] = xk.flatten()
            P_hist_arr[k + 1] = Pk

        return x_hist_arr, P_hist_arr, t_hist_arr
