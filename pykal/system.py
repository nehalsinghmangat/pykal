import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import Union, Optional, Callable, Sequence, List
from pykal.utils.safeio import SafeIO


class System(SafeIO):
    """
    Dynamical System Interface for Simulation and Estimation.

    This class provides a fully validated wrapper for representing continuous-time
    or discrete-time dynamical systems, including state evolution, measurement models,
    and optional process and measurement noise.

    """

    @staticmethod
    def default_h(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x

    @staticmethod
    def default_u() -> NDArray:
        return np.zeros((1, 1))

    def default_process_noise(self) -> NDArray:

        return np.eye(len(self.state_names), len(self.state_names)) * 0.01

    def zero_process_noise(self) -> NDArray:

        return 0 * np.eye(len(self.state_names), len(self.state_names))

    def default_measurement_noise(self) -> NDArray:

        return np.eye(len(self.measurement_names), len(self.measurement_names)) * 0.01

    def zero_measurement_noise(self) -> NDArray:

        return 0 * np.eye(len(self.measurement_names), len(self.measurement_names))

    def _validate_system_type(self, system_type: str) -> str:
        if not isinstance(system_type, str):
            raise TypeError(
                f"System type must be a string, got {type(system_type).__name__}"
            )
        system_type = system_type.lower()
        if system_type not in self.system_types:
            raise ValueError(
                f"Unrecognized system type '{system_type}'. "
                f"Expected one of: {sorted(self.system_types)}"
            )
        return system_type

    system_types = {"cti", "ctv", "dti", "dtv"}

    def __init__(
        self,
        *,
        f: Callable,
        h: Optional[Callable] = None,
        state_names: List[str],
        measurement_names: Optional[List[str]] = None,
        system_type: str = "cti",
        F: Optional[Callable] = None,
        H: Optional[Callable] = None,
        u: Optional[Callable] = None,
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
        input_names: Optional[List[str]] = None,
    ) -> None:

        # Validate string inputs
        self._state_names = self._validate_string_sequence(state_names)
        self._input_names = (
            self._validate_string_sequence(input_names)
            if input_names is not None
            else ["Zero input."]
        )
        self._measurement_names = (
            self._validate_string_sequence(measurement_names)
            if measurement_names is not None
            else state_names
        )
        self._system_type = self._validate_system_type(system_type)

        # Validate function inputs
        if f is None:
            raise TypeError("f cannot be None!")

        self._f = self._validate_func_signature(f)
        self._h = (
            self._validate_func_signature(h) if h is not None else System.default_h
        )
        self._u = (
            self._validate_func_signature(u) if u is not None else System.default_u
        )
        self._Q = (
            self._validate_func_signature(Q)
            if Q is not None
            else self.default_process_noise
        )
        self._R = (
            self._validate_func_signature(R)
            if R is not None
            else self.default_measurement_noise
        )
        self._F = self._validate_func_signature(F) if F is not None else F
        self._H = self._validate_func_signature(H) if H is not None else H

    @property
    def system_type(self):
        return self._system_type

    @system_type.setter
    def system_type(self, system_type):
        self._system_type = self._validate_system_type(system_type)

    # Getters and Setters
    @property
    def state_names(self):
        return self._state_names

    @state_names.setter
    def state_names(self, names):
        self._state_names = self._validate_string_sequence(names)

    @property
    def measurement_names(self):
        return self._measurement_names

    @measurement_names.setter
    def measurement_names(self, names):
        self._measurement_names = self._validate_string_sequence(names)

    @property
    def input_names(self):
        return self._input_names

    @input_names.setter
    def input_names(self, names):
        self._input_names = self._validate_string_sequence(names)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        if f is None:
            raise TypeError("f cannot be None!")
        self._f = self._validate_func_signature(f)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self._h = (
            self._validate_func_signature(h) if h is not None else System.default_h
        )

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        self._u = (
            self._validate_func_signature(u) if u is not None else System.default_u
        )

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        if Q is None:
            self._Q = self.zero_process_noise
        else:
            self._Q = self._validate_func_signature(Q)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        if R is None:
            self._R = self.zero_measurement_noise
        else:
            self._R = self._validate_func_signature(R)

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, F):
        self._F = self._validate_func_signature(F) if F is not None else F

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, H):
        self._H = self._validate_func_signature(H) if H is not None else H

    # Simulators
    def _simulate_states_continuous(
        self,
        *,
        x0: NDArray,
        f: Callable,
        u: Callable,
        Q: Callable,
        t_linspace: NDArray,
        state_names: list,
        input_names: list,
    ) -> tuple[NDArray, NDArray]:

        def wrapped_f(t_k: float, x_flat: NDArray) -> NDArray:
            x_k = x_flat.reshape(-1, 1)
            u_k = self.smart_call(u, x=x_k, t=t_k, expected_shape=(len(input_names), 1))
            dx = self.smart_call(
                f, x=x_k, u=u_k, t=t_k, expected_shape=(len(state_names), 1)
            )
            return dx.flatten()

        t0, tf = t_linspace[0], t_linspace[-1]
        sol = solve_ivp(
            fun=wrapped_f,
            t_span=(t0, tf),
            t_eval=t_linspace,
            y0=x0.flatten(),
            vectorized=False,
        )

        X = sol.y  # shape (n_states, n_steps)
        T = sol.t  # shape (n_steps,)

        for k, t_k in enumerate(T):
            x_k = X[:, k].reshape(-1, 1)
            u_k = self.smart_call(u, x=x_k, t=t_k, expected_shape=(len(input_names), 1))
            Q_k = self.smart_call(
                Q,
                x=x_k,
                u=u_k,
                t=t_k,
                expected_shape=(len(state_names), len(state_names)),
            )
            rk = np.random.multivariate_normal(
                mean=np.zeros(Q_k.shape[0]), cov=Q_k
            ).reshape(-1, 1)
            X[:, k : k + 1] += rk

        return X, T

    def _simulate_states_discrete(
        self,
        *,
        x0: NDArray,
        f: Callable,
        u: Callable,
        Q: Callable,
        t_linspace: NDArray,
        state_names: list,
        input_names: list,
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate a discrete-time state trajectory using forward iteration.

        At each time step, this function:
          1. Evaluates the control input `u(t_k)`
          2. Computes the next state using the user-supplied transition function `f(x, u, t)`
          3. Optionally injects additive Gaussian noise using `Q(x, u, t)`

        All functional inputs are statically validated using
        `safeio.call_validated_function_with_args` to ensure safe dispatch and
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
        safeio.call_validated_function_with_args : Strict functional call utility.
        """

        n = len(state_names)
        n_steps = len(t_linspace) - 1
        X = np.zeros((n, n_steps + 1))
        X[:, 0] = x0.flatten()

        for k in range(n_steps):
            t_k = t_linspace[k]
            x_k = X[:, k].reshape(-1, 1)
            u_k = self.smart_call(u, x=x_k, t=t_k, expected_shape=(len(input_names), 1))

            # Propagate dynamics
            x_next = self.smart_call(
                f, x=x_k, u=u_k, t=t_k, expected_shape=(len(state_names), 1)
            )

            # Process noise
            Q_k = self.smart_call(
                Q,
                x=x_k,
                u=u_k,
                t=t_k,
                expected_shape=(len(state_names), len(state_names)),
            )
            rk = np.random.multivariate_normal(mean=np.zeros(n), cov=Q_k).reshape(-1, 1)

            # Add noise and store
            X[:, k + 1] = (x_next + rk).flatten()

        return X, t_linspace

    def simulate_states(
        self,
        *,
        x0: NDArray,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        override_system_f: Union[Callable, None, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
        override_system_input_names: Union[Sequence[str], bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
    ) -> tuple[NDArray, NDArray]:

        # Override attributes
        state_names = (
            self.state_names
            if override_system_state_names is False
            else self._validate_string_sequence(override_system_state_names)
        )

        input_names = (
            self.input_names
            if override_system_input_names is False
            else self._validate_string_sequence(override_system_input_names)
        )

        # Initial Condition check
        if x0.shape != (len(state_names), 1):
            raise ValueError(
                f"For states {state_names} x0 must have shape ({len(state_names)}, 1), got {x0.shape}"
            )

        t_linspace = self._standardize_time_input_to_linspace(t_vector, t_span, dt)

        # Override methods
        if override_system_f is False:
            f = self.f
        elif override_system_f is None:
            raise TypeError("override_system_f cannot be None!")
        else:
            f = self._validate_func_signature(override_system_f)

        if override_system_u is False:
            u = self.u
        elif override_system_u is None:
            u = System.default_u
        else:
            u = self._validate_func_signature(override_system_u)

        if override_system_Q is False:
            Q = self.Q
        elif override_system_Q is None:
            Q = self.zero_process_noise
        else:
            Q = self._validate_func_signature(override_system_Q)

        if self.system_type in {"cti", "ctv"}:
            return self._simulate_states_continuous(
                x0=x0,
                f=f,
                u=u,
                Q=Q,
                state_names=state_names,
                input_names=input_names,
                t_linspace=t_linspace,
            )

        elif self.system_type in {"dti", "dtv"}:
            return self._simulate_states_discrete(
                x0=x0,
                f=f,
                u=u,
                Q=Q,
                state_names=state_names,
                input_names=input_names,
                t_linspace=t_linspace,
            )

    def simulate_measurements(
        self,
        *,
        X: NDArray,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
        override_system_input_names: Union[Sequence[str], bool] = False,
        override_system_measurement_names: Union[Sequence[str], bool] = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate the system's output measurements based on a known state trajectory.

        For each time step, this function evaluates the measurement model `h(x, u, t)`
        and optionally injects Gaussian measurement noise drawn from `R(x, u, t)`.
        All functions are validated and dispatched through `safeio` to ensure correctness
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

        >>> from pykal.system import System

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

        def return_overriden_params():
            pass

        # Override attributes
        measurement_names = (
            self.measurement_names
            if override_system_measurement_names is False
            else self._validate_string_sequence(override_system_measurement_names)
        )

        input_names = (
            self.input_names
            if override_system_input_names is False
            else self._validate_string_sequence(override_system_input_names)
        )

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_states, n_steps).")

        t_linspace = self._standardize_time_input_to_linspace(t_vector, t_span, dt)

        if X.shape[1] != len(t_linspace):
            raise ValueError("X and T must have the same number of time steps.")
        # Override methods

        h = self._resolve_override(override_system_h, self.h, System.default_h)

        if override_system_h is False:
            h = self.h
        elif override_system_h is None:
            h = System.default_h
        else:
            h = self._validate_func_signature(override_system_h)

        if override_system_u is False:
            u = self.u
        elif override_system_u is None:
            u = System.default_u
        else:
            u = self._validate_func_signature(override_system_u)

        if override_system_R is False:
            R = self.R
        elif override_system_R is None:
            R = self.zero_measurement_noise
        else:
            R = self._validate_func_signature(override_system_R)

        Y = []

        for k, t_k in enumerate(t_linspace):
            x_k = X[:, k].reshape(-1, 1)
            u_k = self.smart_call(u, x=x_k, t=t_k, expected_shape=(len(input_names), 1))
            y_k = self.smart_call(
                h, x=x_k, u=u_k, t=t_k, expected_shape=(len(measurement_names), 1)
            )

            R_k = self.smart_call(
                R,
                x=x_k,
                u=u_k,
                t=t_k,
                expected_shape=(
                    len(measurement_names),
                    len(measurement_names),
                ),
            )
            w_k = np.random.multivariate_normal(
                mean=np.zeros(R_k.shape[0]), cov=R_k
            ).reshape(-1, 1)
            y_k += w_k

            Y.append(y_k.flatten())

        return np.array(Y).T, t_linspace

    # Computations
    def compute_empirical_jacobian_wrt_x(
        self,
        func: Callable,
        epsilon: float = 1e-6,
    ) -> Callable:
        """
        Return a closure that computes the Jacobian ∂f/∂x at a given (x, u, t).

        Parameters
        ----------
        func : Callable
            Function f(x, u, t) returning (n×1) ndarray

        Returns
        -------
        Callable
            A function (xk, uk, tk) -> (n×n) ndarray
        """

        def jacobian_x(xk: NDArray, uk: NDArray, tk: float) -> NDArray:
            Jx, _ = self._compute_empirical_jacobian(func, xk, uk, tk, epsilon=epsilon)
            return Jx

        return jacobian_x

    def compute_empirical_jacobian_wrt_u(
        self,
        func: Callable,
        epsilon: float = 1e-6,
    ) -> Callable:
        """
        Return a closure that computes the Jacobian ∂f/∂u at a given (x, u, t).

        Parameters
        ----------
        func : Callable
            Function f(x, u, t) returning (n×1) ndarray

        Returns
        -------
        Callable
            A function (xk, uk, tk) -> (n×m) ndarray
        """

        def jacobian_u(xk: NDArray, uk: NDArray, tk: float) -> NDArray:
            _, Ju = self._compute_empirical_jacobian(func, xk, uk, tk, epsilon=epsilon)
            return Ju

        return jacobian_u

    def compute_empirical_jacobian_wrt_t(
        self,
        func: Callable,
        epsilon: float = 1e-6,
    ) -> Callable:
        """
        Return a closure that computes the Jacobian ∂f/∂t at a given (x, u, t).

        Parameters
        ----------
        func : Callable
            Function f(x, u, t) returning (n×1) ndarray

        Returns
        -------
        Callable
            A function (xk, uk, tk) -> (n×1) ndarray
        """

        def jacobian_t(xk: NDArray, uk: NDArray, tk: float) -> NDArray:
            _, _, Jt = self._compute_empirical_jacobian(
                func, xk, uk, tk, epsilon=epsilon, include_dt=True
            )
            return Jt

        return jacobian_t

    def _compute_empirical_jacobian(
        self,
        func: Callable,
        xk: NDArray,
        uk: NDArray,
        tk: float,
        epsilon: float = 1e-6,
        include_dt: bool = False,
    ) -> Union[tuple[NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
        """
        Compute empirical Jacobians of a function f(x, u, t) with respect to state, input, and optionally time,
        using central finite differences.

        This is typically used for estimating the Jacobians of a nonlinear **dynamics function** in Kalman filters:
            f : Rⁿ × Rᵐ × R → Rⁿ

        Parameters
        ----------
        func : Callable
            A validated dynamics function of the form f(x, u, t) → Rⁿ, returning a (n, 1) ndarray.
        xk : NDArray
            State vector at time t, of shape (n, 1).
        uk : NDArray
            Input vector at time t, of shape (m, 1).
        tk : float
            Scalar time.
        epsilon : float, optional
            Perturbation size for finite differences. Default is 1e-6.
        include_dt : bool, optional
            Whether to compute the time derivative ∂f/∂t. Default is False.

        Returns
        -------
        Jx : NDArray
            Jacobian ∂f/∂x of shape (n, n).
        Ju : NDArray
            Jacobian ∂f/∂u of shape (n, m).
        Jt : NDArray, optional
            Jacobian ∂f/∂t of shape (n, 1), returned only if `include_dt=True`.

        Examples
        --------
        >>> import numpy as np
        >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return np.array([
        ...         [x[0, 0] + 2 * u[0, 0] + 0.1 * np.sin(t)],
        ...         [x[1, 0] ** 2 + u[0, 0] + t ** 2]
        ...     ])
        >>> x = np.array([[1.0], [2.0]])
        >>> u = np.array([[0.5]])
        >>> Jx, Ju, Jt = Differentiation.compute_empirical_jacobian(f, x, u, tk=1.0, include_dt=True)
        >>> np.round(Jx, 3)
        array([[1., 0.],
               [0., 4.]])
        >>> np.round(Ju, 3)
        array([[2.],
               [1.]])
        >>> np.round(Jt, 3)
        array([[0.054],
               [2.   ]])
        """
        # Evaluate once to get output dimension (n_output,)
        f_base = self.smart_call(func, x=xk, u=uk, t=tk)
        n_output = f_base.shape[0]
        n_states = xk.shape[0]
        n_inputs = uk.shape[0]

        Jx = np.zeros((n_output, n_states))
        Ju = np.zeros((n_output, n_inputs))
        Jt = np.zeros((n_output, 1))  # <-- fix: must match output shape, not state dim!

        # ∂f/∂x
        for i in range(n_states):
            dx = np.zeros_like(xk)
            dx[i, 0] = epsilon
            f_plus = self.smart_call(
                func, x=xk + dx, u=uk, t=tk, expected_shape=(n_output, 1)
            )
            f_minus = self.smart_call(
                func, x=xk - dx, u=uk, t=tk, expected_shape=(n_output, 1)
            )
            Jx[:, i : i + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂u
        for j in range(n_inputs):
            du = np.zeros_like(uk)
            du[j, 0] = epsilon
            f_plus = self.smart_call(
                func, x=xk, u=uk + du, t=tk, expected_shape=(n_output, 1)
            )
            f_minus = self.smart_call(
                func, x=xk, u=uk - du, t=tk, expected_shape=(n_output, 1)
            )
            Ju[:, j : j + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂t
        if include_dt:
            f_plus = self.smart_call(
                func, x=xk, u=uk, t=tk + epsilon, expected_shape=(n_output, 1)
            )
            f_minus = self.smart_call(
                func, x=xk, u=uk, t=tk - epsilon, expected_shape=(n_output, 1)
            )
            Jt = (f_plus - f_minus) / (2 * epsilon)
            return Jx, Ju, Jt

        return Jx, Ju

    def compute_observability_matrix(
        self,
        *,
        x0: NDArray,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        override_system_f: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
        override_system_input_names: Union[Sequence[str], bool] = False,
        override_system_measurement_names: Union[Sequence[str], bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
        epsilon: float = 1e-5,
    ) -> NDArray:
        """
        Compute the empirical observability matrix by stacking finite differences
        of the output trajectories with respect to each perturbed state.

        Parameters
        ----------
        x0 : NDArray
            Initial state vector, shape (n, 1).
        t_span : tuple[float, float], optional
            Time interval [t0, tf] over which to simulate.
        dt : float, optional
            Time step used for simulations.
        t_vector : NDArray, optional
            Explicit time vector to override t_span/dt.
        override_system_u : Callable or bool, optional
            Control input function u(t), or False to use system default.
        override_system_h : Callable or bool, optional
            Measurement function h(x, u, t), or False to use system default.
        override_system_input_names : Sequence[str] or bool, optional
            Input names to override the system's input_names.
        override_system_measurement_names : Sequence[str] or bool, optional
            Measurement names to override the system's measurement_names.
        epsilon : float
            Perturbation magnitude for state deviations.

        Returns
        -------
        O : NDArray
            Empirical observability matrix of shape ((K * m), n)
            where K is the number of time steps and m is the number of measurements.
        """
        measurement_names = (
            self.measurement_names
            if override_system_measurement_names is False
            else self._validate_string_sequence(override_system_measurement_names)
        )
        input_names = (
            self.input_names
            if override_system_input_names is False
            else self._validate_string_sequence(override_system_input_names)
        )

        t_linspace = self._standardize_time_input_to_linspace(t_vector, t_span, dt)
        n_states = len(self.state_names)
        m_meas = len(measurement_names)
        K = len(t_linspace)

        # (K * m, n) matrix
        O = np.zeros((K * m_meas, n_states))

        for i in range(n_states):
            ei = np.zeros_like(x0, dtype=float)
            ei[i, 0] = epsilon
            x_plus = x0 + ei
            x_minus = x0 - ei

            X_plus, _ = self.simulate_states(
                x0=x_plus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=override_system_state_names,
                override_system_input_names=override_system_input_names,
                override_system_Q=override_system_Q,
            )

            X_minus, _ = self.simulate_states(
                x0=x_minus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=override_system_state_names,
                override_system_input_names=override_system_input_names,
                override_system_Q=override_system_Q,
            )

            Y_plus, _ = self.simulate_measurements(
                X=X_plus,
                t_vector=t_linspace,
                override_system_h=override_system_h,
                override_system_u=override_system_u,
                override_system_measurement_names=measurement_names,
                override_system_input_names=input_names,
                override_system_R=override_system_R,
            )
            Y_minus, _ = self.simulate_measurements(
                X=X_minus,
                t_vector=t_linspace,
                override_system_h=override_system_h,
                override_system_u=override_system_u,
                override_system_measurement_names=measurement_names,
                override_system_input_names=input_names,
                override_system_R=override_system_R,
            )

            delta_Y = (Y_plus - Y_minus) / (2 * epsilon)  # shape (m, K)

            # Reshape column-wise: [dy(t_0); dy(t_1); ...] ∈ ℝ^{K·m}
            O[:, i] = delta_Y.T.flatten()

        return O

    def compute_observability_of_system_from_grammian(
        self,
        *,
        x0: NDArray,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_input_names: Union[Sequence[str], bool] = False,
        override_system_measurement_names: Union[Sequence[str], bool] = False,
        epsilon: float = 1e-5,
    ) -> NDArray:
        r"""
        Estimate the empirical observability Gramian using central finite differences
        of the output trajectories with respect to state perturbations.

        This captures the aggregate sensitivity of the output measurements to small
        perturbations in each state, over a time window [t0, tf].

        Empirical Gramian:
            .. math::
                W_O = \frac{1}{2\epsilon^2} \sum_{i=1}^n
                    \int_{t_0}^{t_f} (y_i^+(t) - y_i^-(t))(y_i^+(t) - y_i^-(t))^\top dt

        where :math:`y_i^+(t)` and :math:`y_i^-(t)` are the measurement trajectories
        from initial conditions :math:`x_0 \pm \epsilon e_i`.

        Parameters
        ----------
        h : Callable
            Measurement function h(x, u, t), returning (m, 1) array.
        x0 : NDArray
            Initial state vector, shape (n, 1).
        u : Callable
            Input function u(t) returning (m, 1).
        t_span : tuple[float, float]
            Time interval [t0, tf] over which to simulate.
        dt : float
            Time step used for simulations.
        epsilon : float, optional
            Perturbation magnitude for state deviations.

        Returns
        -------
        W : NDArray
            Empirical observability Gramian, shape (n, n).

        Examples
        --------
        >>> x0 = np.array([[1.0], [0.0]])
        >>> observer = Observer()  # Must provide .simulate_measurements
        >>> W.shape
        (2, 2)
        """

        O = self.compute_observability_matrix(
            x0=x0,
            t_span=t_span,
            dt=dt,
            t_vector=t_vector,
            override_system_u=override_system_u,
            override_system_h=override_system_h,
            override_system_input_names=override_system_input_names,
            override_system_measurement_names=override_system_measurement_names,
            epsilon=epsilon,
        )
        return O.T @ O

    def compute_CRB_of_system_from_grammian(self, W: NDArray) -> dict[str, float]:
        """
        Compute the Cramér-Rao Bound (CRB) diagonal values for each state
        from the observability Gramian.

        Parameters
        ----------
        W : NDArray
            Empirical observability Gramian, shape (n, n)

        Returns
        -------
        dict[str, float]
            Dictionary mapping state name → CRB diagonal value (scalar).
        """
        ridge = 1e-5  # regularization for numerical stability
        CRB = np.linalg.inv(W + ridge * np.eye(W.shape[0]))
        return dict(zip(self.state_names, np.diag(CRB)))

    def compute_observability_of_states_from_grammian_nullspace(
        self,
        *,
        W: NDArray,
        tol: float = 1e-10,
    ) -> dict[str, float]:
        """
        Compute per-state observability indices using the nullspace projection method.

        Parameters
        ----------
        W : NDArray
            Empirical observability Gramian, shape (n, n).
        state_names : list of str, optional
            Optional list of state variable names. Defaults to x0, x1, ..., xn-1.
        tol : float
            Threshold below which eigenvalues are considered "zero" (nullspace).

        Returns
        -------
        dict[str, float]
            Dictionary mapping state name to observability index ∈ [0, 1].

        W = observer.compute_empirical_observability_grammian(x0, t_span=(0, 10), dt=0.1)
        obs_idx = compute_state_observability_indices(W, state_names=system.state_names)
        print(obs_idx)
        {'position': 0.98, 'velocity': 0.12, 'bias': 0.001}

        """
        n = W.shape[0]
        eigvals, eigvecs = np.linalg.eigh(W)

        # Identify eigenvectors corresponding to (near-)nullspace
        nullspace_mask = eigvals < tol
        null_vectors = eigvecs[:, nullspace_mask]  # shape (n, r)

        state_names = self.state_names

        indices = {}
        for j in range(n):
            e_j = np.zeros((n, 1))
            e_j[j, 0] = 1.0
            proj = null_vectors.T @ e_j  # shape (r, 1)
            norm2 = np.sum(proj**2)  # squared projection
            indices[state_names[j]] = float(1.0 - norm2)

        return indices

    def compute_observability_of_states_over_time_from_metric(
        self,
        *,
        x0: NDArray,
        observability_metric_of_states: Callable,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        divide_time_into_k_windows: Optional[int] = None,
        window_length_in_points: Optional[int] = None,
        window_length_in_time: Optional[float] = None,
        overlap_points_between_windows: Optional[int] = None,
        overlap_time_between_windows: Optional[float] = None,
    ):
        t_linspace = self._standardize_time_input_to_linspace(t_vector, t_span, dt)
        list_of_t_windows = self._standardize_make_time_windows(
            t_linspace,
            divide_time_into_k_windows=divide_time_into_k_windows,
            window_length_in_points=window_length_in_points,
            window_length_in_time=window_length_in_time,
            overlap_points_between_windows=overlap_points_between_windows,
            overlap_time_between_windows=overlap_time_between_windows,
        )

        scores_over_time = []
        for window_t in list_of_t_windows:
            scores = observability_metric_of_states(x0=x0, t_vector=window_t)
            scores_over_time.append(scores)

        return scores_over_time, list_of_t_windows

    def compute_eigenvalues_of_observability_of_system_metric():
        pass

    def compute_CN_of_observability_of_system_metric():
        pass

    def compute_observability_of_system_from_CRLB():
        pass

    def compute_observability_of_states_from_CRLB():
        pass

    def compute_observability_of_states_from_P(
        self,
        *,
        P_series: pd.Series,
        t_eval: float,
        normalize: bool = True,
        clip: bool = True,
        override_system_state_names: Union[Sequence[str], bool] = False,
        eps: float = 1e-12,
    ) -> dict[str, float]:
        """
        Compute stochastic observability scores per state from the posterior covariance
        matrix P(t), using the inverse diagonal method from Ramos et al. (2021).

        Parameters
        ----------
        P_series : pd.Series
            A time-indexed Series of (n x n) covariance matrices from a Kalman filter run.
        t_eval : float
            Time at which to evaluate the observability scores.
        normalize : bool, default True
            Whether to normalize scores so they sum to 1.
        clip : bool, default True
            Whether to clip scores to [0, 1] after normalization.
        eps : float, default 1e-12
            Small constant to prevent division by zero.

        Returns
        -------
        dict[str, float]
            Mapping from state name to stochastic observability score.
        """
        if not isinstance(P_series, pd.Series):
            raise TypeError("P_series must be a pandas Series indexed by time")

        try:
            t_closest = P_series.index.get_indexer([t_eval], method="nearest")[0]
        except Exception as e:
            raise ValueError(f"Failed to locate nearest time to t={t_eval}: {e}")

        P = P_series.iloc[t_closest]

        if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("Each value in P_series must be a square NumPy array")

        diag = np.diag(P)
        inv_diag = 1.0 / (diag + eps)  # avoid div by zero

        if normalize:
            inv_diag = inv_diag / np.sum(inv_diag)

        if clip:
            inv_diag = np.clip(inv_diag, 0.0, 1.0)

        state_names = (
            self.state_names
            if override_system_state_names is False
            else self._validate_string_sequence(override_system_state_names)
        )

        if len(inv_diag) != len(state_names):
            raise ValueError(
                "Length of P diagonal does not match number of state names"
            )

        return dict(zip(self.state_names, inv_diag))

    def compute_error_pointwise(
        self,
        truedf: pd.DataFrame,
        estdf: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute pointwise RMSE, MAE, and MaxErr between true and estimated states.

        Parameters
        ----------
        truedf : pd.DataFrame
            Ground truth state values.
        estdf : pd.DataFrame
            Estimated state values.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['RMSE', 'MAE', 'MaxErr'].

        Examples
        --------
        >>> truedf = pd.DataFrame({'x0': [1.0, 2.0], 'x1': [1.0, 3.0]}, index=[0.0, 1.0])
        >>> estdf = pd.DataFrame({'x0': [1.1, 1.9], 'x1': [0.9, 3.2]}, index=[0.0, 1.0])
        >>> from pykal.utils.utils_computation import Error
        >>> Error.compute_pointwise_error_metrics(truedf, estdf).round(2)
              RMSE  MAE  MaxErr
        0.0  0.10  0.10     0.1
        1.0  0.16  0.15     0.2
        """
        if not truedf.index.equals(estdf.index):
            raise ValueError("Indices of true and estimated DataFrames must match.")

        errors = (estdf[truedf.columns] - truedf).values  # shape (T, n)

        rmse = np.sqrt(np.mean(errors**2, axis=1))
        mae = np.mean(np.abs(errors), axis=1)
        maxerr = np.max(np.abs(errors), axis=1)

        return pd.DataFrame(
            {"RMSE": rmse, "MAE": mae, "MaxErr": maxerr}, index=truedf.index
        )

    def compute_error_nees(
        self,
        truedf: pd.DataFrame,
        estdf: pd.DataFrame,
        P_seq: pd.Series,
        tol: float = 1e-9,
    ) -> pd.Series:
        """
        Compute the Normalized Estimation Error Squared (NEES) over time.

        Parameters
        ----------
        truedf : pd.DataFrame
            True state values indexed by time.
        estdf : pd.DataFrame
            Estimated state values indexed by time.
        P_seq : pd.Series of np.ndarray
            Covariance matrices (n × n), one per time step, indexed by time.
        tol : float, optional
            Small diagonal regularization added to P_k for numerical stability.

        Returns
        -------
        pd.Series
            NEES values indexed by time.

        Raises
        ------
        ValueError
            If indices do not match or if P_seq entries are not 2D arrays.

        Examples
        --------
        >>> import pandas as pd
        >>> from pykal.utils.utils_computation import Error
        >>> truedf = pd.DataFrame({'x0': [1.0], 'x1': [1.0]}, index=[0.0])
        >>> estdf = pd.DataFrame({'x0': [1.1], 'x1': [0.9]}, index=[0.0])
        >>> P_seq = pd.Series({0.0: 0.1 * np.eye(2)})
        >>> Error.compute_nees_only(truedf, estdf, P_seq).round(2)
        0.0    0.2
        Name: NEES, dtype: float64
        """
        if not isinstance(P_seq, pd.Series):
            raise TypeError("P_seq must be a pandas Series of covariance matrices.")

        if not truedf.index.equals(estdf.index) or not truedf.index.equals(P_seq.index):
            raise ValueError("Indices of truedf, estdf, and P_seq must all match.")

        nees_values = []

        for t in truedf.index:
            x_true = truedf.loc[t].values.reshape(-1, 1)
            x_est = estdf.loc[t].values.reshape(-1, 1)
            e_k = x_est - x_true

            P_k = P_seq.loc[t]
            if not isinstance(P_k, np.ndarray) or P_k.ndim != 2:
                raise ValueError(f"P_seq[{t}] must be a 2D NumPy array.")

            P_k_reg = P_k + tol * np.eye(P_k.shape[0])

            try:
                nees_k = float(e_k.T @ np.linalg.inv(P_k_reg) @ e_k)
            except np.linalg.LinAlgError:
                nees_k = np.nan

            nees_values.append(nees_k)

        return pd.Series(nees_values, index=truedf.index, name="NEES")

    def compute_error_nll(
        self,
        truedf: pd.DataFrame,
        estdf: pd.DataFrame,
        P_seq: pd.Series,
        tol: float = 1e-9,
    ) -> pd.Series:
        """
        Compute the Negative Log-Likelihood (NLL) over time under Gaussian assumption.

        Parameters
        ----------
        truedf : pd.DataFrame
            True state values.
        estdf : pd.DataFrame
            Estimated state values.
        P_seq : pd.Series of np.ndarray
            Covariance matrices per timestep.
        tol : float, optional
            Diagonal regularization.

        Returns
        -------
        pd.Series
            Negative log-likelihood values indexed by time.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pykal.utils.utils_computation import Error
        >>> truedf = pd.DataFrame({'x0': [1.0], 'x1': [2.0]}, index=[0.0])
        >>> estdf = pd.DataFrame({'x0': [1.1], 'x1': [1.9]}, index=[0.0])
        >>> P_seq = pd.Series({0.0: 0.1 * np.eye(2)})
        >>> Error.compute_nll(truedf, estdf, P_seq).round(2)
        0.0    -0.36
        Name: NLL, dtype: float64
        """
        if not truedf.index.equals(estdf.index) or not truedf.index.equals(P_seq.index):
            raise ValueError("Indices must match for truedf, estdf, and P_seq.")

        nll_vals = []

        for t in truedf.index:
            x_true = truedf.loc[t].values.reshape(-1, 1)
            x_est = estdf.loc[t].values.reshape(-1, 1)
            e_k = x_est - x_true
            P_k = P_seq.loc[t] + tol * np.eye(len(e_k))

            try:
                inv_P = np.linalg.inv(P_k)
                nees_k = float(e_k.T @ inv_P @ e_k)
                logdet = np.linalg.slogdet(2 * np.pi * P_k)[1]
                nll_k = 0.5 * (logdet + nees_k)
            except np.linalg.LinAlgError:
                nll_k = np.nan

            nll_vals.append(nll_k)

        return pd.Series(nll_vals, index=truedf.index, name="NLL")

    def compute_error_mse_per_state(
        self, truedf: pd.DataFrame, estdf: pd.DataFrame
    ) -> pd.Series:
        """
        Compute mean squared error per state over all time steps.

        Returns
        -------
        pd.Series
            MSE per state (averaged over time).

        Examples
        --------
        >>> import pandas as pd
        >>> from pykal.utils.utils_computation import Error
        >>> truedf = pd.DataFrame({'x0': [1.0, 2.0], 'x1': [2.0, 4.0]}, index=[0.0, 1.0])
        >>> estdf = pd.DataFrame({'x0': [1.1, 1.9], 'x1': [2.1, 3.8]}, index=[0.0, 1.0])
        >>> Error.compute_mse_per_state(truedf, estdf).round(4)
        x0    0.010
        x1    0.025
        dtype: float64
        """
        if not truedf.index.equals(estdf.index):
            raise ValueError("Indices must match.")

        diff = estdf[truedf.columns] - truedf
        return (diff**2).mean()

    @staticmethod
    def combine_data_and_time_into_DataFrame(
        data: NDArray, time: NDArray, column_names: List[str]
    ) -> pd.DataFrame:
        """
        Combine a 2D data array and a 1D time array into a pandas DataFrame.

        Parameters
        ----------
        data : NDArray
            Array of shape (n_variables, n_time_steps).
        time : NDArray
            Array of shape (n_time_steps,) or (n_time_steps, 1).
        column_names : list
            List of length n_variables giving column names for the data.

        Returns
        -------
        df : pd.DataFrame
            A DataFrame with time as the index and variable names as columns.

        Raises
        ------
        ValueError
            If shapes of data, time, or column_names are incompatible.
        TypeError
            If input types are incorrect.

        Examples
        --------
        >>> data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        >>> time = np.array([0.1, 0.2, 0.3])
        >>> column_names = ["x0", "x1"]
        >>> df = SystemIO.combine_data_and_time_into_DataFrame(data, time, column_names)
        >>> df
              x0    x1
        time
        0.1   1.0  10.0
        0.2   2.0  20.0
        0.3   3.0  30.0

        >>> bad_data = np.array([1.0, 2.0, 3.0])
        >>> SystemIO.combine_data_and_time_into_DataFrame(bad_data, time, column_names)
        Traceback (most recent call last):
            ...
        ValueError: `data` must be a 2D array, got shape (3,).

        >>> time_wrong = np.array([0.1, 0.2])
        >>> SystemIO.combine_data_and_time_into_DataFrame(data, time_wrong, column_names)
        Traceback (most recent call last):
            ...
        ValueError: Time length 2 does not match number of time steps in data 3.

        >>> bad_names = ["x0"]
        >>> SystemIO.combine_data_and_time_into_DataFrame(data, time, bad_names)
        Traceback (most recent call last):
            ...
        ValueError: Length of column_names (1) does not match number of variables in data (2).

        >>> not_a_list = "x0,x1"
        >>> SystemIO.combine_data_and_time_into_DataFrame(data, time, not_a_list)
        Traceback (most recent call last):
            ...
        TypeError: `column_names` must be a list, got <class 'str'>

        >>> time_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> SystemIO.combine_data_and_time_into_DataFrame(data, time_matrix, column_names)
        Traceback (most recent call last):
            ...
        ValueError: `time` should be of shape (n_steps,) or (n_steps, 1), got (2, 3).

        >>> SystemIO.combine_data_and_time_into_DataFrame(data, time, tuple(column_names))
        Traceback (most recent call last):
            ...
        TypeError: `column_names` must be a list, got <class 'tuple'>

        """

        if not isinstance(data, np.ndarray):
            raise TypeError(f"`data` must be an np.ndarray, got {type(data)}")
        if not isinstance(time, np.ndarray):
            raise TypeError(f"`time` must be an np.ndarray, got {type(time)}")
        if not isinstance(column_names, list):
            raise TypeError(f"`column_names` must be a list, got {type(column_names)}")

        if data.ndim != 2:
            raise ValueError(f"`data` must be a 2D array, got shape {data.shape}.")

        n_vars, n_steps = data.shape

        if time.ndim == 2:
            if time.shape[1] != 1:
                raise ValueError(
                    f"`time` should be of shape (n_steps,) or (n_steps, 1), got {time.shape}."
                )
            time = time.flatten()
        elif time.ndim != 1:
            raise ValueError(
                f"`time` must be 1D or 2D with one column, got shape {time.shape}."
            )

        if time.shape[0] != n_steps:
            raise ValueError(
                f"Time length {time.shape[0]} does not match number of time steps in data {n_steps}."
            )

        if len(column_names) != n_vars:
            raise ValueError(
                f"Length of column_names ({len(column_names)}) does not match number of variables in data ({n_vars})."
            )

        df = pd.DataFrame(data.T, index=time, columns=column_names)
        df.index.name = "time"
        return df

    @staticmethod
    def combine_matrix_series_with_time(data: NDArray, time: NDArray) -> pd.Series:
        """
        Combine a 3D array (n x n x m) with a 1D or (m, 1) time array into a pd.Series
        indexed by time, where each entry is an (n x n) array.

        Parameters
        ----------
        data : NDArray
            Array of shape (n, n, m), representing a time-indexed matrix series.
        time : NDArray
            Time array of shape (m,) or (m, 1) matching the third dimension of `data`.

        Returns
        -------
        pd.Series
            A Series indexed by time with each element being an (n x n) NDArray.

        Raises
        ------
        ValueError
            If data is not 3D or time shape is incompatible.

        Examples
        --------
        >>> covariances = np.stack([np.eye(2), 2 * np.eye(2), 3 * np.eye(2)], axis=2)
        >>> time = np.array([0.0, 1.0, 2.0])
        >>> series = SystemIO.combine_matrix_series_with_time(covariances, time)
        >>> series.iloc[0]
        array([[1., 0.],
               [0., 1.]])
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"`data` must be a NumPy array, got {type(data)}")
        if data.ndim != 3:
            raise ValueError(
                f"`data` must be a 3D array of shape (n, n, m), got shape {data.shape}"
            )

        n1, n2, m = data.shape
        if n1 != n2:
            raise ValueError(
                f"`data` must be square in the first two dimensions, got shape ({n1}, {n2}, {m})"
            )

        if not isinstance(time, np.ndarray):
            raise TypeError(f"`time` must be a NumPy array, got {type(time)}")
        if time.ndim == 2 and time.shape[1] == 1:
            time = time.flatten()
        elif time.ndim != 1:
            raise ValueError(
                f"`time` must be a 1D array or shape (m, 1), got shape {time.shape}"
            )

        if len(time) != m:
            raise ValueError(
                f"Length of `time` ({len(time)}) does not match number of matrices ({m})"
            )

        return pd.Series({t: data[:, :, i] for i, t in enumerate(time)})
