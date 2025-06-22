import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import Callable, Optional, List, Union
from utils.iosafety import SystemIO as ios, SafeIO as safeio, SystemType


class System:
    """
    Dynamical System Interface for Simulation and Estimation.

    This class provides a fully validated wrapper for representing continuous-time
    or discrete-time dynamical systems, including state evolution, measurement models,
    and optional process and measurement noise.

    Key Features
    ------------
    - Dynamical function `f(x, u, t)` and measurement model `h(x, u, t)`
    - Input function `u(t)`, and optional process/measurement noise models `Q(x, u, t)`, `R(x, u, t)`
    - Explicit time structure via `SystemType` enum
    - Full static validation of function signatures and shapes via `SystemIO` and `SafeIO`
    - Robust simulation methods for state and measurement trajectories with support for noise injection
    - Override mechanisms for dynamic substitution of model components

    All user-supplied functions are passed through decorators that:
      • Enforce proper parameter names (`x`, `u`, `t` and their aliases)
      • Require correct type annotations (especially `NDArray`)
      • Verify return shapes match expected dimensions
      • Raise early, informative errors on violation

    Parameters
    ----------
    f : Callable
        State transition function `f(x, u, t)` or `f(x, u)`.
    h : Callable
        Measurement function `h(x, u, t)` or `h(x, u)`.
    state_names : list of str
        Names of each state dimension (length n).
    measurement_names : list of str
        Names of each measurement dimension (length m).
    system_type : SystemType, optional
        Specifies whether the system is continuous/discrete and time-varying/invariant.
        Defaults to `SystemType.CONTINUOUS_TIME_INVARIANT`.
    u : Callable or None, optional
        Optional input function `u(t)`; if `None`, defaults to zero input.
    Q : Callable or None, optional
        Optional process noise covariance function `Q(x, u, t)`; if `None`, defaults to zeros.
    R : Callable or None, optional
        Optional measurement noise covariance function `R(x, u, t)`; if `None`, defaults to zeros.
    input_names : list of str, optional
        Names of input channels. Used only for documentation and validation.

    Attributes
    ----------
    f, h, Q, R, u : Callable
        Validated system functions and noise models.
    state_names : list of str
        Ordered names of state dimensions.
    measurement_names : list of str
        Ordered names of measurement dimensions.
    input_names : list of str or None
        Ordered names of input dimensions (if specified).
    system_type : SystemType
        Timing convention of the system.

    Methods
    -------
    simulate_states(...)
        Simulates the system's state trajectory for a given initial condition and time grid.
    simulate_measurements(...)
        Simulates measurement data corresponding to a given state trajectory.

    Notes
    -----
    The `System` class ensures that invalid inputs or unexpected function behavior
    are caught early through strict decorator-based validation. Any failure to comply
    with expected function signatures or return shapes will raise informative `TypeError`
    or `ValueError` exceptions at assignment time.

    The simulation routines support injection of Gaussian noise and allow temporary
    overrides of system functions for what-if analysis, hybrid modeling, or uncertainty
    quantification.

    See Also
    --------
    SystemType : Enum of timing structures.
    SystemIO : Static validation utilities for setting functions and names.
    SafeIO : Core function signature/type enforcement logic.
    EKFIO : Input checker class for Kalman filter applications.
    """

    def __init__(
        self,
        *,
        f: Callable,
        h: Callable,
        state_names: List[str],
        measurement_names: List[str],
        system_type: SystemType = SystemType.CONTINUOUS_TIME_INVARIANT,
        u: Optional[Callable] = None,
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
        input_names: Optional[List[str]] = None,
    ) -> None:

        self._f = ios.set_f(f)
        self._h = ios.set_h(h)
        self._u = ios.set_u(u)
        self._Q = ios.set_Q(Q, state_names)
        self._R = ios.set_R(R, measurement_names)
        self._state_names = ios.set_state_names(state_names)
        self._measurement_names = ios.set_measurement_names(measurement_names)
        self._input_names = ios.set_input_names(input_names)
        self._system_type = ios.set_system_type(system_type)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, func):
        self._f = ios.set_f(func)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, func):
        self._h = ios.set_h(func)

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, func):
        self._Q = ios.set_Q(func, self._state_names)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, func):
        self._R = ios.set_R(func, self._measurement_names)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, func):
        self._u = ios.set_u(func)

    @property
    def state_names(self):
        return self._state_names

    @state_names.setter
    def state_names(self, names):
        self._state_names = ios.set_state_names(names)

    @property
    def measurement_names(self):
        return self._measurement_names

    @measurement_names.setter
    def measurement_names(self, names):
        self._measurement_names = ios.set_measurement_names(names)

    @property
    def input_names(self):
        return self._input_names

    @input_names.setter
    def input_names(self, names):
        self._input_names = ios.set_input_names(names)

    @property
    def system_type(self):
        return self._system_type

    @system_type.setter
    def system_type(self, val):
        self._system_type = ios.set_system_type(val)

    @staticmethod
    def _simulate_states_continuous(
        x0: NDArray,
        f: Callable,
        u: Callable,
        Q: Callable,
        dt: float,
        t_span: tuple,
        t_eval: NDArray,
        process_noise: bool,
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate a continuous-time state trajectory using numerical integration.

        This method uses `scipy.integrate.solve_ivp` to solve the differential system
        defined by the dynamics function `f(x, u, t)`, with optional injection of
        Gaussian process noise from `Q(x, u, t)` at each timestep.

        All input functions must conform to the statically validated signatures enforced
        via `SafeIO.call_validated_function_with_args`, ensuring correct shape, typing,
        and parameter names.

        Parameters
        ----------
        x0 : NDArray
            Initial state vector of shape (n_states, 1).
        f : Callable
            Dynamics function, typically of the form `f(x, u, t) -> NDArray`.
        u : Callable
            Input function of the form `u(t) -> NDArray`.
        Q : Callable
            Process noise covariance function, `Q(x, u, t) -> NDArray` (n x n).
        dt : float
            Time step used to construct `t_eval` if not explicitly provided.
        t_span : tuple
            Time interval `(t0, tf)` over which to simulate.
        t_eval : NDArray
            Evaluation times for the solver. If not provided, must be constructed externally.
        process_noise : bool
            If True, inject Gaussian noise sampled from `Q(x, u, t)` at each time point.

        Returns
        -------
        X : NDArray
            Simulated state trajectory, shape `(n_states, n_steps)`.
        T : NDArray
            Time vector associated with the simulation, shape `(n_steps,)`.

        Notes
        -----
        - The ODE is solved in flattened form (`x.flatten()`), but reshaped internally
          for correct matrix operations and validation.
        - Noise is injected *after* deterministic integration, as additive perturbations
          at each timepoint.
        - All function calls (`f`, `u`, `Q`) are dispatched using `SafeIO.call_validated_function_with_args`
          to enforce safety and suppress silent failure from malformed input functions.

        See Also
        --------
        simulate_states : High-level unified simulation method.
        SystemType : Enum that distinguishes continuous from discrete systems.
        SafeIO.call_validated_function_with_args : Strict dispatch utility.
        """

        def wrapped_f(t_k: float, x_flat: NDArray) -> NDArray:
            x_k = x_flat.reshape(-1, 1)
            u_k = u(t_k).reshape(-1, 1)
            dx = safeio.call_validated_function_with_args(f, x=x_k, u=u_k, t=t_k)
            return dx.flatten()

        sol = solve_ivp(
            fun=wrapped_f,
            t_span=t_span,
            y0=x0.flatten(),
            t_eval=t_eval,
            vectorized=False,
        )

        X = sol.y  # shape (n_states, n_steps)
        T = sol.t

        if process_noise is True:
            for k, t_k in enumerate(T):
                x_k = X[:, k].reshape(-1, 1)
                u_k = u(t_k).reshape(-1, 1)
                Q_k = safeio.call_validated_function_with_args(Q, x=x_k, u=u_k, t=t_k)
                noise = np.random.multivariate_normal(
                    mean=np.zeros(Q_k.shape[0]), cov=Q_k
                ).reshape(-1, 1)
                X[:, k : k + 1] += noise

        return X, T

    @staticmethod
    def _simulate_states_discrete(
        x0: NDArray,
        f: Callable,
        u: Callable,
        Q: Callable,
        dt: float,
        n_steps: int,
        t0: float,
        process_noise: bool,
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate a discrete-time state trajectory using forward iteration.

        At each time step, this function:
          1. Evaluates the control input `u(t_k)`
          2. Computes the next state using the user-supplied transition function `f(x, u, t)`
          3. Optionally injects additive Gaussian noise using `Q(x, u, t)`

        All functional inputs are statically validated using
        `SafeIO.call_validated_function_with_args` to ensure safe dispatch and
        suppress silent shape/type failures.

        Parameters
        ----------
        x0 : NDArray
            Initial state vector, shape `(n_states, 1)`.
        f : Callable
            Discrete-time dynamics function: `f(x, u, t) -> NDArray`.
        u : Callable
            Input function: `u(t) -> NDArray`.
        Q : Callable
            Process noise covariance function: `Q(x, u, t) -> NDArray` (n x n).
        dt : float
            Fixed time step interval between updates.
        n_steps : int
            Number of simulation steps to run.
        t0 : float
            Initial simulation time.
        process_noise : bool
            If True, draws Gaussian noise at each step using `Q(x, u, t)`.

        Returns
        -------
        X : NDArray
            Simulated state trajectory, shape `(n_states, n_steps + 1)`.
            Includes initial state `x0` at `X[:, 0]`.
        T : NDArray
            Time vector, shape `(n_steps + 1,)`, constructed as
            `T[k] = t0 + k * dt`.

        Notes
        -----
        - Noise is added after each state transition, i.e., `x_{k+1} ← f(...) + ε`.
        - Each call to `f`, `u`, and `Q` is verified against the system interface specification.
        - Time is explicitly incremented; this method does not rely on numerical integration.

        See Also
        --------
        simulate_states : High-level unified simulation interface.
        SystemType : Enum of discrete vs. continuous system types.
        SafeIO.call_validated_function_with_args : Strict functional call utility.
        """

        n = x0.shape[0]
        X = np.zeros((n, n_steps + 1))
        X[:, 0] = x0.flatten()

        T = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)

        for k in range(n_steps):
            t_k = T[k]
            x_k = X[:, k].reshape(-1, 1)
            u_k = u(t_k).reshape(-1, 1)
            x_next = safeio.call_validated_function_with_args(f, x=x_k, u=u_k, t=t_k)

            if process_noise:
                Q_k = safeio.call_validated_function_with_args(Q, x=x_k, u=u_k, t=t_k)
                noise = np.random.multivariate_normal(
                    mean=np.zeros(n), cov=Q_k
                ).reshape(-1, 1)
                x_next += noise

            X[:, k + 1] = x_next.flatten()

        return X, T

    def simulate_states(
        self,
        *,
        x0: NDArray,
        dt: float = 1.0,
        t_span: tuple = (0, 10),
        t_eval: Optional[NDArray] = None,
        n_steps: Optional[int] = 100,
        process_noise: bool = True,
        override_system_f: Union[Callable, None, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate a state trajectory over time using the system’s dynamic model.

        This method dispatches to either a continuous-time integrator or a
        discrete-time simulator depending on the system type. Functional overrides
        allow experimentation with alternate dynamics, inputs, and noise models at runtime.

        All inputs are validated for shape, type, and dimension safety. Simulation
        functions are dispatched through `SafeIO.call_validated_function_with_args`
        to ensure that type and argument mismatches are caught early.

        Parameters
        ----------
        x0 : NDArray
            Initial state vector of shape `(n_states, 1)`.
        dt : float, optional
            Integration step size (for continuous systems) or iteration interval (discrete).
        t_span : tuple of float, optional
            Time interval `(t0, tf)` over which to simulate.
        t_eval : NDArray, optional
            Evaluation times for continuous systems. If `None`, uniform spacing using `dt` is used.
        n_steps : int, optional
            Number of steps for discrete-time systems. Ignored for continuous systems unless needed.
        process_noise : bool, optional
            If `True`, inject Gaussian process noise at each step using `Q(x, u, t)`.

        override_system_f : Union[Callable, None, bool], optional
            - `False` (default): use internal system dynamics `f`.
            - `None`: raise an error — `f` is required.
            - Callable: use this function as dynamics override.

        override_system_u : Union[Callable, None, bool], optional
            - `False` (default): use internal input function `u`.
            - `None`: treat as zero input.
            - Callable: use this function as input override.

        override_system_Q : Union[Callable, None, bool], optional
            - `False` (default): use internal process noise model `Q`.
            - `None`: disable process noise.
            - Callable: use this function as noise override.

        Returns
        -------
        X : NDArray
            Simulated state trajectory, shape `(n_states, n_steps)` for discrete,
            or `(n_states, len(t_eval))` for continuous.
        T : NDArray
            Time vector used in simulation, shape `(n_steps,)` or matching `t_eval`.

        Raises
        ------
        TypeError
            If inputs have invalid types or missing shape constraints.
        ValueError
            If input dimensions are incompatible or the system type is not recognized.

        Notes
        -----
        - Uses `solve_ivp` for continuous-time systems with vectorized input disabled.
        - Supports runtime substitution of `f`, `u`, or `Q` with full input validation.
        - The internal system type determines which simulation branch is called.

        See Also
        --------
        _simulate_states_continuous : Low-level integrator for continuous systems.
        _simulate_states_discrete : Manual loop-based simulator for discrete systems.
        simulate_measurements : Measurement simulator for use with state outputs.
        SystemType : Enum defining timing structure.

        Examples
        --------
        >>> import numpy as np

        >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return np.array([[x[1, 0]], [-x[0, 0]]])  # Simple harmonic oscillator

        >>> def h(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x[:1]  # Observe only position

        >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.01 * np.eye(2)

        >>> def R(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.1 * np.eye(1)

        >>> def u(t: float) -> NDArray:
        ...     return np.zeros((1, 1))  # No control input

        >>> sys = System(
        ...     f=f,
        ...     h=h,
        ...     Q=Q,
        ...     R=R,
        ...     u=u,
        ...     state_names=["x0", "x1"],
        ...     measurement_names=["x0"],
        ...     system_type=SystemType.CONTINUOUS_TIME_INVARIANT,
        ... )

        >>> x0 = np.array([[1.0], [0.0]])
        >>> X, T = sys.simulate_states(x0=x0, dt=0.1, t_span=(0, 1), process_noise=False)
        >>> X.shape[0] == 2 and X.shape[1] == len(T)
        True
        """

        n_states = len(self.state_names)

        if not isinstance(x0, np.ndarray):
            raise TypeError("x0 must be a NumPy ndarray.")
        if x0.shape != (n_states, 1):
            raise ValueError(f"x0 must have shape ({n_states}, 1), got {x0.shape}")

        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be a float or int.")
        if not (isinstance(t_span, tuple) and len(t_span) == 2):
            raise TypeError("t_span must be a tuple of length 2.")
        if t_eval is not None:
            if not isinstance(t_eval, np.ndarray):
                raise TypeError("t_eval must be a NumPy ndarray if provided.")
            if t_eval.ndim != 1:
                raise ValueError("t_eval must be a 1D array.")
        if n_steps is not None and not isinstance(n_steps, int):
            raise TypeError("n_steps must be an integer if provided.")
        if not isinstance(process_noise, bool):
            raise TypeError("process_noise must be a boolean.")

        f = self._f if override_system_f is False else ios.set_f(override_system_f)
        u = self._u if override_system_u is False else ios.set_u(override_system_u)
        Q = (
            self._Q
            if override_system_Q is False
            else ios.set_Q(override_system_Q, self.state_names)
        )

        if self.system_type in {
            SystemType.CONTINUOUS_TIME_INVARIANT,
            SystemType.CONTINUOUS_TIME_VARYING,
        }:
            if t_eval is None:
                t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
            return System._simulate_states_continuous(
                x0,
                f,
                u,
                Q,
                dt,
                t_span,
                t_eval,
                process_noise,
            )

        elif self.system_type in {
            SystemType.DISCRETE_TIME_INVARIANT,
            SystemType.DISCRETE_TIME_VARYING,
        }:
            if n_steps is None:
                n_steps = int((t_span[1] - t_span[0]) / dt)
            return System._simulate_states_discrete(
                x0,
                f,
                u,
                Q,
                dt,
                n_steps,
                t0=t_span[0],
                process_noise=process_noise,
            )

        else:
            raise ValueError(f"Unsupported system type: {self.system_type}")

    def simulate_measurements(
        self,
        *,
        X: NDArray,
        T: NDArray,
        measurement_noise: bool = True,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate the system's output measurements based on a known state trajectory.

        For each time step, this function evaluates the measurement model `h(x, u, t)`
        and optionally injects Gaussian measurement noise drawn from `R(x, u, t)`.
        All functions are validated and dispatched through `SafeIO` to ensure correctness
        and early failure for shape/type mismatches.

        Parameters
        ----------
        X : NDArray
            Simulated state trajectory of shape `(n_states, n_steps)`.
        T : NDArray
            Time vector of shape `(n_steps,)`, corresponding to each column in `X`.
        measurement_noise : bool, optional
            If `True`, inject Gaussian measurement noise at each time step using `R(x, u, t)`.

        override_system_u : Union[Callable, None, bool], optional
            - `False` (default): use the system’s internal input function `u(t)`.
            - `None`: treat as zero input.
            - Callable: use this function as an override for `u(t)`.

        override_system_h : Union[Callable, None, bool], optional
            - `False` (default): use the system’s internal measurement function `h(x, u, t)`.
            - `None`: raises an error — `h` is required.
            - Callable: use this function as an override for `h`.

        override_system_R : Union[Callable, None, bool], optional
            - `False` (default): use the system’s internal measurement noise function `R(x, u, t)`.
            - `None`: treat measurements as noiseless.
            - Callable: use this function as a noise override.

        Returns
        -------
        Y : NDArray
            Simulated measurement trajectory, of shape `(n_measurements, n_steps)`.
        T : NDArray
            Same time vector as input, returned unchanged.

        Raises
        ------
        TypeError
            If `X` or `T` are not NumPy arrays.
        ValueError
            If time and state vectors do not align in length.

        Notes
        -----
        - All simulation components (`h`, `R`, `u`) can be replaced at runtime.
        - Overrides are passed through the same validation decorators as the system constructor.
        - Ensures strong type and shape guarantees across the full simulation loop.

        See Also
        --------
        simulate_states : Simulates the underlying state evolution for the system.
        set_h, set_R, set_u : Functional registration and validation utilities.

        Examples
        --------
        >>> import numpy as np


        >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return np.array([[x[1, 0]], [-x[0, 0]]])  # Harmonic oscillator

        >>> def h(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x[:1]  # Observe position

        >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.01 * np.eye(2)

        >>> def R(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.1 * np.eye(1)

        >>> def u(t: float) -> NDArray:
        ...     return np.zeros((1, 1))

        >>> sys = System(
        ...     f=f,
        ...     h=h,
        ...     Q=Q,
        ...     R=R,
        ...     u=u,
        ...     state_names=["x0", "x1"],
        ...     measurement_names=["x0"],
        ...     system_type=SystemType.CONTINUOUS_TIME_INVARIANT,
        ... )

        >>> x0 = np.array([[1.0], [0.0]])
        >>> X, T = sys.simulate_states(x0=x0, dt=0.1, t_span=(0, 1), process_noise=False)
        >>> Y, T_out = sys.simulate_measurements(X=X, T=T, measurement_noise=False)
        >>> Y.shape[0] == 1 and Y.shape[1] == X.shape[1]
        True

        """

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray.")
        if not isinstance(T, np.ndarray):
            raise TypeError("T must be a NumPy ndarray.")
        if X.shape[1] != T.shape[0]:
            raise ValueError("X and T must have the same number of time steps.")

        h = self._h if override_system_h is False else ios.set_h(override_system_h)
        R = (
            self._R
            if override_system_R is False
            else ios.set_R(override_system_R, self.measurement_names)
        )
        u = self._u if override_system_u is False else ios.set_u(override_system_u)

        Y = []

        for k, t_k in enumerate(T):
            x_k = X[:, k].reshape(-1, 1)
            u_k = u(t_k).reshape(-1, 1)
            y_k = safeio.call_validated_function_with_args(h, x=x_k, u=u_k, t=t_k)

            if measurement_noise:
                R_k = safeio.call_validated_function_with_args(R, x=x_k, u=u_k, t=t_k)
                noise = np.random.multivariate_normal(
                    mean=np.zeros(R_k.shape[0]), cov=R_k
                ).reshape(-1, 1)
                y_k += noise

            Y.append(y_k.flatten())

        return np.array(Y).T, T
