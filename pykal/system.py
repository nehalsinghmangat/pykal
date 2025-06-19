import warnings
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from typing import Callable, Optional, Tuple
from utils.func_io_restrict import (
    validate_system_function,
    call_validated_function_with_args,
)
from enum import Enum, auto


class SystemType(Enum):
    r"""
    Enumeration of timing assumptions for plant / measurement models.

    Members
    -------
    CONTINUOUS_TIME_INVARIANT
        .. math::
           \dot x = f(x, u)
        (no explicit time argument)

    CONTINUOUS_TIME_VARYING
        .. math::
           \dot x = f(x, u, t)

    DISCRETE_TIME_INVARIANT
        .. math::
           x_{k+1} = f(x_k, u_k)

    DISCRETE_TIME_VARYING
        .. math::
           x_{k+1} = f(x_k, u_k, t_k)

    This enum exists purely for type-safety and clarity when specifying
    your system's timing assumptions.
    """

    CONTINUOUS_TIME_INVARIANT = auto()
    CONTINUOUS_TIME_VARYING = auto()
    DISCRETE_TIME_INVARIANT = auto()
    DISCRETE_TIME_VARYING = auto()


class System:
    """
    Representation of a dynamical system for use in simulation and state estimation.

    This class encapsulates:
      - System dynamics (`f`) and measurement models (`h`)
      - Process and measurement noise models (`Q`, `R`)
      - Timing structure (`system_type`)
      - Named state and measurement dimensions

    All functions are statically validated to conform to expected signatures using
    `validate_system_function`. Supported parameter aliases are defined via:
      - `_get_alias_for_x()` for state
      - `_get_alias_for_u()` for input
      - `_get_alias_for_t()` for time

    Parameters
    ----------
    f : Callable
        System dynamics function: should return the time derivative of the state (for continuous systems)
        or the next state (for discrete systems).
    h : Callable
        Measurement function: maps the state to measurements.
    Q : Callable
        Process noise covariance function.
    R : Callable
        Measurement noise covariance function.
    state_names : list of str
        Names of each state variable (for labeling and indexing).
    measurement_names : list of str
        Names of each measurement dimension.
    system_type : SystemType, optional
        Enum indicating system timing: continuous/discrete, time-varying or invariant.

    Attributes
    ----------
    f, h, Q, R : Callable
        Dynamical, measurement, and noise models.
    state_names : list
        Names of state variables.
    measurement_names : list
        Names of measurement outputs.
    system_type : SystemType
        Defines timing structure.

    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: Callable,
        R: Callable,
        state_names: list,
        measurement_names: list,
        system_type: SystemType = SystemType.CONTINUOUS_TIME_INVARIANT,
    ) -> None:

        self.f = validate_system_function(f)
        self.h = validate_system_function(h)
        self.Q = validate_system_function(Q)
        self.R = validate_system_function(R)
        self.system_type = system_type
        self.state_names = state_names
        self.measurement_names = measurement_names

    def simulate_states_continuous(
        self,
        x0: NDArray,
        u: Optional[Callable] = None,
        dt: float = 1.0,
        t_span: tuple = (0, 10),
        t_eval: Optional[NDArray] = None,
        process_noise: bool = True,
    ):
        """
            Simulate continuous-time states using `solve_ivp`.

        Parameters
            ----------
            x0 : NDArray
            Initial state (n, 1)
            u : Callable, optional
            Input function of time (default: zero)
            dt : float
            Time step used if `t_eval` not provided
            t_span : tuple
            Interval (t0, tf)
            t_eval : array-like, optional
            Times to evaluate solution
            process_noise : bool
            Whether to add Gaussian noise from `Q`

        Returns
            -------
            X : NDArray
            Simulated states (n_states, n_steps)
            T : NDArray
            Time vector (n_steps,)
        """
        if u is None:
            u = lambda t: np.zeros_like(x0)
            if t_eval is None:
                t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

        def wrapped_f(t_k, x_flat):
            x_k = x_flat.reshape(-1, 1)
            u_k = u(t_k).reshape(-1, 1)
            dx = call_validated_function_with_args(self.f, x=x_k, u=u_k, t=t_k)
            return dx.flatten()

        sol = solve_ivp(
            fun=wrapped_f,
            t_span=t_span,
            y0=x0.flatten(),
            t_eval=t_eval,
            vectorized=False,
        )

        X = sol.y  # (n_states, n_steps)
        T = sol.t
        if process_noise and self.Q is not None:
            for k, t_k in enumerate(T):
                Q_k = call_validated_function_with_args(
                    self.Q, x=X[:, k].reshape(-1, 1), u=u(t_k), t=t_k
                )
                noise = np.random.multivariate_normal(
                    mean=np.zeros(Q_k.shape[0]), cov=Q_k
                ).reshape(-1, 1)
                X[:, k : k + 1] += noise

        return X, T

    def simulate_states_discrete(
        self,
        x0: NDArray,
        u: Optional[Callable] = None,
        dt: float = 1.0,
        n_steps: int = 100,
        t0: float = 0.0,
        process_noise: bool = True,
    ):
        """
            Simulate discrete-time system state evolution using manual iteration.

        Parameters
            ----------
            x0 : NDArray
            Initial state of shape (n, 1)
            u : Callable, optional
            Input function of time (default: zero input)
            dt : float
            Time step duration
            n_steps : int
            Number of simulation steps
            t0 : float
            Initial time
            process_noise : bool
            Whether to inject Gaussian noise using Q

        Returns
            -------
            X : NDArray
            Simulated state trajectory, shape (n_states, n_steps + 1)
            T : NDArray
            Discrete time vector, shape (n_steps + 1,)
        """
        if u is None:
            u = lambda t: np.zeros_like(x0)

        n = x0.shape[0]
        X = np.zeros((n, n_steps + 1))
        X[:, 0] = x0.flatten()

        T = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)

        for k in range(n_steps):
            t_k = T[k]
            x_k = X[:, k].reshape(-1, 1)
            u_k = u(t_k).reshape(-1, 1)

            x_next = call_validated_function_with_args(self.f, x=x_k, u=u_k, t=t_k)

            if process_noise and self.Q is not None:
                Q_k = call_validated_function_with_args(self.Q, x=x_k, u=u_k, t=t_k)
                noise = np.random.multivariate_normal(
                    mean=np.zeros(Q_k.shape[0]), cov=Q_k
                ).reshape(-1, 1)
                x_next += noise

                X[:, k + 1] = x_next.flatten()

        return X, T

    def simulate_states(
        self,
        x0: NDArray,
        u: Optional[Callable] = None,
        dt: float = 1.0,
        t_span: tuple = (0, 10),
        t_eval: Optional[NDArray] = None,
        n_steps: Optional[int] = None,
        process_noise: bool = True,
    ):
        """
        Unified interface for simulating states (continuous or discrete based on system_type).

        Parameters
        ----------
        x0 : NDArray
            Initial state (n, 1)
        u : Callable, optional
            Input function of time
        dt : float
            Time step
        t_span : tuple
            Time interval (only used for continuous)
        t_eval : array-like or None
            Evaluation times (continuous only)
        n_steps : int, optional
            Number of steps (required for discrete if t_eval not provided)
        process_noise : bool
            Whether to add process noise using Q

        Returns
        -------
        X : NDArray
            State trajectory (n_states, n_steps or len(t_eval))
        T : NDArray
            Time vector

        Examples
        --------
        Continuous-Time Invariant System:
        >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x + u
        >>> def h(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x
        >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.01 * np.eye(x.shape[0])
        >>> def R(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.1 * np.eye(x.shape[0])

        >>> x0 = np.zeros((2, 1))
        >>> u_fn = lambda t: np.array([[0.1], [0.0]])
        >>> sys_cti = System(f, h, Q, R, ["x0", "x1"], ["x0", "x1"], SystemType.CONTINUOUS_TIME_INVARIANT)
        >>> X, T = sys_cti.simulate_states(x0=x0, u=u_fn, dt=0.1, t_span=(0, 1))
        >>> X.shape[0] == 2 and len(T) == X.shape[1]
        True

        Discrete-Time Invariant System:
        >>> def f_d(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x + 0.1 * u
        >>> sys_dti = System(f_d, h, Q, R, ["x0", "x1"], ["x0", "x1"], SystemType.DISCRETE_TIME_INVARIANT)
        >>> X, T = sys_dti.simulate_states(x0=x0, u=u_fn, dt=0.1, n_steps=10)
        >>> X.shape[0] == 2 and len(T) == X.shape[1]
        True

        Continuous-Time Varying System:
        >>> def f_ctv(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return (1 + 0.1 * t) * (x + u)
        >>> def h_ctv(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x - 0.05 * t
        >>> sys_ctv = System(f_ctv, h_ctv, Q, R, ["x0", "x1"], ["x0", "x1"], SystemType.CONTINUOUS_TIME_VARYING)
        >>> X, T = sys_ctv.simulate_states(x0=x0, u=u_fn, dt=0.1, t_span=(0, 1))
        >>> X.shape[0] == 2 and len(T) == X.shape[1]
        True

        Discrete-Time Varying System:
        >>> def f_dtv(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x + (0.05 + 0.01 * t) * u
        >>> def h_dtv(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x + 0.01 * t
        >>> sys_dtv = System(f_dtv, h_dtv, Q, R, ["x0", "x1"], ["x0", "x1"], SystemType.DISCRETE_TIME_VARYING)
        >>> X, T = sys_dtv.simulate_states(x0=x0, u=u_fn, dt=0.1, n_steps=10)
        >>> X.shape[0] == 2 and len(T) == X.shape[1]
        True
        """
        if self.system_type in {
            SystemType.CONTINUOUS_TIME_INVARIANT,
            SystemType.CONTINUOUS_TIME_VARYING,
        }:
            return self.simulate_states_continuous(
                x0=x0,
                u=u,
                dt=dt,
                t_span=t_span,
                t_eval=t_eval,
                process_noise=process_noise,
            )
        elif self.system_type in {
            SystemType.DISCRETE_TIME_INVARIANT,
            SystemType.DISCRETE_TIME_VARYING,
        }:
            if n_steps is None:
                n_steps = int((t_span[1] - t_span[0]) / dt)
            return self.simulate_states_discrete(
                x0=x0,
                u=u,
                dt=dt,
                n_steps=n_steps,
                t0=t_span[0],
                process_noise=process_noise,
            )

        else:
            raise ValueError(f"Unsupported system type: {self.system_type}")

    def simulate_measurements(
        self,
        X: NDArray,
        T: NDArray,
        u: Optional[Callable] = None,
        measurement_noise: bool = True,
    ):
        """
                Generate measurements from simulated state trajectories with optional noise.

                 Parameters
                 ----------
                 X : NDArray
                 Simulated states, shape (n_states, n_steps)
                 T : NDArray
                 Time vector, shape (n_steps,)
                 u : Callable, optional
                 Input function of time (default: zero input)
                 measurement_noise : bool
                 Whether to inject Gaussian noise using R

                Returns
                -------
                 Y : NDArray
                 Measurements, shape (n_measurements, n_steps)
                 T : NDArray
                 Time vector (same as input)

                Examples
                --------
        Continuous-Time Invariant System:
                >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
                ...     return x + u
                >>> def h(x: NDArray, u: NDArray, t: float) -> NDArray:
                ...     return x
                >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
                ...     return 0.01 * np.eye(x.shape[0])
                >>> def R(x: NDArray, u: NDArray, t: float) -> NDArray:
                ...     return 0.1 * np.eye(x.shape[0])

                >>> x0 = np.zeros((2, 1))
                >>> u_fn = lambda t: np.array([[0.1], [0.0]])
                >>> sys_cti = System(f, h, Q, R, ["x0", "x1"], ["x0", "x1"], SystemType.CONTINUOUS_TIME_INVARIANT)
                >>> X, T = sys_cti.simulate_states(x0=x0, u=u_fn, dt=0.1, t_span=(0, 1))
                >>> X.shape[0] == 2 and len(T) == X.shape[1]
                True
                >>> Y, T2 = sys_cti.simulate_measurements(X=X, T=T, u=u_fn, measurement_noise=False)
                >>> Y.shape == (2, len(T)) and np.allclose(Y, X)
                True
        """
        if u is None:
            u = lambda t: np.zeros((X.shape[0], 1))

        Y = []

        for k, t_k in enumerate(T):
            x_k = X[:, k].reshape(-1, 1)
            u_k = u(t_k).reshape(-1, 1)

            y_k = call_validated_function_with_args(self.h, x=x_k, u=u_k, t=t_k)

            if measurement_noise and self.R is not None:
                R_k = call_validated_function_with_args(self.R, x=x_k, u=u_k, t=t_k)
                noise = np.random.multivariate_normal(
                    mean=np.zeros(R_k.shape[0]), cov=R_k
                ).reshape(-1, 1)
                y_k += noise

            Y.append(y_k.flatten())

        return np.array(Y).T, T
