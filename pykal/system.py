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

    system_types = {"cti", "ctv", "dti", "dtv"}

    @staticmethod
    def default_h(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x

    @staticmethod
    def default_u(input_names: Sequence[str]) -> Callable:
        def u() -> NDArray:
            return np.zeros((len(input_names), 1))

        return u

    @staticmethod
    def default_Q(state_names: Sequence[str]) -> Callable:
        def Q() -> NDArray:
            return np.eye(len(state_names), len(state_names)) * 0.01

        return Q

    @staticmethod
    def zero_Q(state_names: Sequence[str]) -> Callable:
        def Q() -> NDArray:
            return np.eye(len(state_names), len(state_names)) * 0.0

        return Q

    @staticmethod
    def default_R(measurement_names: Sequence[str]) -> Callable:

        def R() -> NDArray:
            return np.eye(len(measurement_names), len(measurement_names)) * 0.01

        return R

    @staticmethod
    def zero_R(measurement_names: Sequence[str]) -> Callable:

        def R() -> NDArray:
            return np.eye(len(measurement_names), len(measurement_names)) * 0.0

        return R

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
            self._validate_func_signature(u)
            if u is not None
            else System.default_u(self.input_names)
        )
        self._Q = (
            self._validate_func_signature(Q)
            if Q is not None
            else self.default_Q(self.state_names)
        )
        self._R = (
            self._validate_func_signature(R)
            if R is not None
            else self.default_R(self.measurement_names)
        )
        self._F = (
            self._validate_func_signature(F)
            if F is not None
            else self.compute_jacobian_wrt_x(self.f)
        )
        self._H = (
            self._validate_func_signature(H)
            if H is not None
            else self.compute_jacobian_wrt_x(self.h)
        )

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
            self._validate_func_signature(u)
            if u is not None
            else System.default_u(self.input_names)
        )

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        if Q is None:
            self._Q = self.zero_Q(self.state_names)
        else:
            self._Q = self._validate_func_signature(Q)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        if R is None:
            self._R = self.zero_R
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
        output_df: Optional[bool] = False,
        override_system_f: Union[Callable, None, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_state_names: Union[List[str], bool] = False,
        override_system_input_names: Union[List[str], bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
    ) -> Union[pd.DataFrame, tuple[NDArray, NDArray]]:

        state_names = self._resolve_override_str_list(
            override_system_state_names, self.state_names
        )

        input_names = self._resolve_override_str_list(
            override_system_input_names, self.input_names
        )

        f = self._resolve_override_func(override_system_f, self.f)

        u = self._resolve_override_func(
            override_system_u, self.u, System.default_u(input_names)
        )

        if x0.shape != (len(state_names), 1):
            raise ValueError(
                f"For states {state_names} x0 must have shape ({len(state_names)}, 1), got {x0.shape}"
            )

        t_linspace = self._standardize_time_input_to_linspace(t_vector, t_span, dt)

        Q = self._resolve_override_func(
            override_system_Q, self.Q, System.zero_Q(state_names)
        )

        if self.system_type in {"cti", "ctv"}:
            X, T = self._simulate_states_continuous(
                x0=x0,
                f=f,
                u=u,
                Q=Q,
                state_names=state_names,
                input_names=input_names,
                t_linspace=t_linspace,
            )

        elif self.system_type in {"dti", "dtv"}:
            X, T = self._simulate_states_discrete(
                x0=x0,
                f=f,
                u=u,
                Q=Q,
                state_names=state_names,
                input_names=input_names,
                t_linspace=t_linspace,
            )

        if output_df:
            X_df = pd.DataFrame(X.T, index=T, columns=state_names)
            X_df.index.name = "time"
            return X_df
        else:
            return X, T

    def simulate_measurements(
        self,
        *,
        X: Optional[NDArray] = None,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        X_df: Optional[pd.DataFrame] = None,
        input_df: Optional[bool] = False,
        output_df: Optional[bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
        override_system_input_names: Union[Sequence[str], bool] = False,
        override_system_measurement_names: Union[Sequence[str], bool] = False,
    ) -> Union[pd.DataFrame, tuple[NDArray, NDArray]]:
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

        # Convert from DataFrame if needed
        if input_df:
            if X_df is None:
                raise ValueError("X_df must be provided when input_df=True")
            T = X_df.index.to_numpy()
            X = X_df.to_numpy().T  # shape (n_states, n_steps)
        else:
            if X is None:
                raise ValueError("X must be provided when input_df=False")
            if not isinstance(X, np.ndarray):
                raise TypeError("X must be a NumPy array")

            T = self._standardize_time_input_to_linspace(t_vector, t_span, dt)
            if X.shape[1] != len(T):
                raise ValueError(
                    f"X has {X.shape[1]} steps, but time vector has {len(T)} steps"
                )

        measurement_names = self._resolve_override_str_list(
            override_system_measurement_names, self.measurement_names
        )

        input_names = self._resolve_override_str_list(
            override_system_input_names, self.input_names
        )

        h = self._resolve_override_func(override_system_h, self.h, System.default_h)

        u = self._resolve_override_func(
            override_system_u, self.u, System.default_u(input_names)
        )

        R = self._resolve_override_func(
            override_system_R, self.R, System.zero_R(measurement_names)
        )

        Y = []

        for k, t_k in enumerate(T):
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

        if output_df:
            Y_df = pd.DataFrame(np.array(Y), index=T, columns=measurement_names)
            Y_df.index.name = "time"
            return Y_df
        else:
            return np.array(Y).T, T

    def _matrix_jacobian_wrt_x_u_t(
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

    def compute_jacobian_wrt_x(
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
            Jx, _ = self._matrix_jacobian_wrt_x_u_t(func, xk, uk, tk, epsilon=epsilon)
            return Jx

        return jacobian_x

    def compute_jacobian_wrt_u(
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
            _, Ju = self._matrix_jacobian_wrt_x_u_t(func, xk, uk, tk, epsilon=epsilon)
            return Ju

        return jacobian_u

    def compute_jacobian_wrt_t(
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
            _, _, Jt = self._matrix_jacobian_wrt_x_u_t(
                func, xk, uk, tk, epsilon=epsilon, include_dt=True
            )
            return Jt

        return jacobian_t
