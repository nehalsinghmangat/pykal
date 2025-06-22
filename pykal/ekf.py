from _base_kf import BaseKF
from system import System, SystemType
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from typing import List
from utils.iosafety import SafeIO as safeio, EKFIO as eio
from system import System


class EKF(BaseKF):
    """
    Extended Kalman Filter (EKF) implementation.

    This class implements the EKF for nonlinear state estimation by linearizing
    the system dynamics and measurement models about the current estimate. The
    filter propagates the state mean and covariance using first-order Jacobians
    provided at runtime.

    Inherits From
    -------------
    BaseKF
        Abstract base class defining the interface for all Kalman Filter variants.

    Parameters
    ----------
    sys : System
        A validated `System` object describing the system dynamics, measurement
        model, noise models, and timing structure. Must provide callable functions:
          - f(x, u, t): state dynamics
          - h(x, u, t): measurement model
          - Q(x, u, t): process noise covariance
          - R(x, u, t): measurement noise covariance
        The system must also define the system type (e.g., discrete/continuous)
        and variable name metadata for states and measurements.

    Notes
    -----
    The EKF uses externally provided Jacobian functions for the dynamics and
    measurement models:
      - F(x, u, t) ≈ ∂f/∂x
      - H(x, u, t) ≈ ∂h/∂x

    These are required at runtime in the `run()` method. All model evaluations,
    including f, h, Q, R, F, and H, are dispatched through safe validated calls
    using `call_validated_function_with_args`, ensuring input consistency and
    error isolation.

    The EKF supports both continuous-time and discrete-time system variants.
    """

    def __init__(self, sys: System):
        super().__init__(sys)

    def _predict(
        self,
        xk: NDArray,
        Pk: NDArray,
        Fk: NDArray,
        Qk: NDArray,
        dt: float,
        uk: NDArray,
        tk: float,
    ) -> tuple[NDArray, NDArray]:
        r"""
        Internal EKF prediction step (time update).

        Propagates the mean and covariance of the state estimate forward in time
        using the system's dynamics and linearized model. Handles both discrete-time
        and continuous-time systems by checking the `SystemType`.

        For continuous-time systems, Euler integration is used:
        \[
        x_{k+1} = x_k + f(x_k, u_k, t_k)\,\Delta t,
        \quad
        P_{k+1} = P_k + (F_k P_k + P_k F_k^\top + Q_k)\,\Delta t
        \]

        For discrete-time systems, direct transition is used:
        \[
        x_{k+1} = f(x_k, u_k, t_k),
        \quad
        P_{k+1} = F_k P_k F_k^\top + Q_k
        \]

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

        Returns
        -------
        x_pred : NDArray
            Predicted state estimate after propagation, shape (n, 1).
        P_pred : NDArray
            Predicted covariance after propagation, shape (n, n).
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
            P_pred = Fk @ Pk @ Fk.T + Qk

        return x_pred, P_pred

    def _update(
        self,
        xk: NDArray,
        Pk: NDArray,
        yk: NDArray,
        Hk: NDArray,
        Rk: NDArray,
        dt: float,
        uk: Optional[NDArray],
        tk: Optional[float],
    ) -> tuple[NDArray, NDArray]:
        r"""
        Internal EKF measurement update step (correction).

        Incorporates a new measurement into the current state and covariance estimate
        using the standard EKF innovation-based update equations.

        The Kalman gain is computed as:
        \[
        K_k = P_k H_k^\top (H_k P_k H_k^\top + R_k)^{-1}
        \]

        The updated state and covariance are given by:
        \[
        \begin{aligned}
        \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k (y_k - h(\hat{x}_{k|k-1}, u_k, t_k)) \\\\
        P_{k|k} &= (I - K_k H_k) P_k (I - K_k H_k)^\top + K_k R_k K_k^\top
        \end{aligned}
        \]

        A small ridge term is added to the innovation covariance to ensure numerical
        stability and prevent inversion errors.

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

        Returns
        -------
        x_upd : NDArray
            Updated state estimate after incorporating the measurement, shape (n, 1).
        P_upd : NDArray
            Updated covariance matrix, shape (n, n).
        """

        Sk = Hk @ Pk @ Hk.T + Rk
        ridge = 1e-9 * np.eye(Sk.shape[0])
        try:
            Sk_inv = np.linalg.inv(Sk + ridge)
        except np.linalg.LinAlgError:
            Sk_inv = np.linalg.pinv(Sk + ridge)

        Kk = Pk @ Hk.T @ Sk_inv
        y_pred = safeio.call_validated_function_with_args(self.sys.h, x=xk, u=uk, t=tk)
        x_upd = xk + Kk @ (yk - y_pred)
        I_KH = np.eye(Pk.shape[0]) - Kk @ Hk
        P_upd = I_KH @ Pk @ I_KH.T + Kk @ Rk @ Kk.T

        return x_upd, P_upd

    @eio.verify_run_inputs
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
        u: Callable,
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

        Returns
        -------
        x_hist : NDArray
            Filtered state estimates at all times, shape (T+1, n).
        P_hist : NDArray
            Covariance estimates at all times, shape (T+1, n, n).
        t_hist : NDArray
            Time vector corresponding to each state estimate, shape (T+1,).

        Examples
        --------
        >>> import numpy as np

        >>> from numpy.typing import NDArray

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

        >>> sys = System(
        ...     f=f, h=h, Q=Q, R=R, u=u,
        ...     state_names=["x0", "x1"],
        ...     measurement_names=["x0"],
        ...     system_type=SystemType.DISCRETE_TIME_INVARIANT,
        ... )

        >>> ekf = EKF(sys)

        >>> x0 = np.array([[0.0], [1.0]])
        >>> P0 = 0.01 * np.eye(2)

        >>> # Generate synthetic measurements
        >>> X_sim, T_sim = sys.simulate_states(x0=x0, dt=0.1, t_span=(0, 1.0), process_noise=False)
        >>> Y, _ = sys.simulate_measurements(X=X_sim, T=T_sim, measurement_noise=False)

        >>> X_est, P_est, T_out = ekf.run(
        ...     x0=x0, P0=P0, Y=Y.T,
        ...     F=F, H=H,
        ...     start_time=0.0,
        ...     dt=0.1,
        ...     u=u,
        ... )

        >>> X_est.shape[0] == Y.shape[1] + 1
        True

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

        for k in range(N_steps):
            yk = Y[k].reshape(-1, 1)
            tk = t_hist_arr[k]
            uk = safeio.call_validated_function_with_args(u, t=tk)

            Fk = safeio.call_validated_function_with_args(F, x=xk, u=uk, t=tk)
            Hk = safeio.call_validated_function_with_args(H, x=xk, u=uk, t=tk)
            Qk = safeio.call_validated_function_with_args(self.sys.Q, x=xk, u=uk, t=tk)
            Rk = safeio.call_validated_function_with_args(self.sys.R, x=xk, u=uk, t=tk)

            xk, Pk = self._predict(xk, Pk, Fk, Qk, dt, uk, tk)
            xk, Pk = self._update(xk, Pk, yk, Hk, Rk, dt, uk, tk)

            x_hist_arr[k + 1] = xk.flatten()
            P_hist_arr[k + 1] = Pk

        return x_hist_arr, P_hist_arr, t_hist_arr
