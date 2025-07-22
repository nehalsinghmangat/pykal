import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Union, Optional, Callable, Sequence, List, Tuple
from numpy.linalg import cholesky, inv, LinAlgError, trace, eigh
from scipy.integrate import solve_ivp
from pykal_core.control_system.system import System


class Simulation:
    # Simulators
    @classmethod
    def _simulate_states_continuous(
        cls,
        sys: System,
        *,
        x0: NDArray,
        f: Callable,
        u: Callable,
        Q: Callable,
        t_linspace: NDArray,
        state_names: list,
    ) -> tuple[NDArray, NDArray]:

        def wrapped_f(t_k: float, x_flat: NDArray) -> NDArray:
            x_k = x_flat.reshape(-1, 1)
            u_k = sys.safeio.smart_call(u, x=x_k, t=t_k)
            dx = sys.safeio.smart_call(
                f, x=x_k, u=u_k, t=t_k, expected_shape=(len(state_names), 1)
            )
            # if Value error setting an array element with a sequence, you probably fucked up your array, maybe make sure accessing [0][0]
            # also, should check to make sure passed function wrapped has sa valid return and also if npthing happens you are probably none
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
            u_k = sys.safeio.smart_call(u, x=x_k, t=t_k)
            Q_k = sys.safeio.smart_call(
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

    @classmethod
    def _simulate_states_discrete(
        cls,
        sys: System,
        *,
        x0: NDArray,
        f: Callable,
        u: Callable,
        Q: Callable,
        t_linspace: NDArray,
        state_names: list,
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
            u_k = sys.safeio.smart_call(u, x=x_k, t=t_k)

            # Propagate dynamics
            x_next = sys.safeio.smart_call(
                f, x=x_k, u=u_k, t=t_k, expected_shape=(len(state_names), 1)
            )

            # Process noise
            Q_k = sys.safeio.smart_call(
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

    @classmethod
    def of_state_trajectories(
        cls,
        sys: System,
        *,
        x0: NDArray,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        output_df: Optional[bool] = False,
        signal: Optional[Callable] = None,
        override_system_f: Union[Callable, None, bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
    ) -> Union[pd.DataFrame, tuple[NDArray, NDArray]]:

        state_names = sys.state_names

        if x0.shape != (len(state_names), 1):
            raise ValueError(
                f"For states {state_names} x0 must have shape ({len(state_names)}, 1), got {x0.shape}"
            )

        u = sys.make_u_zero()

        if signal:
            u = signal

        f = sys.safeio._resolve_override_func(override_system_f, sys.f, sys.f_zero)

        Q = sys.safeio._resolve_override_func(
            override_system_Q, sys.Q, sys.make_Q(state_names, multiply_Q_by_scalar=0)
        )

        t_linspace = sys.safeio._standardize_time_input_to_linspace(
            t_vector, t_span, dt
        )

        if sys.system_type in {"cti", "ctv"}:
            X, T = cls._simulate_states_continuous(
                sys=sys,
                x0=x0,
                f=f,
                u=u,
                Q=Q,
                state_names=state_names,
                t_linspace=t_linspace,
            )

        elif sys.system_type in {"dti", "dtv"}:
            X, T = cls._simulate_states_discrete(
                sys=sys,
                x0=x0,
                f=f,
                u=u,
                Q=Q,
                state_names=state_names,
                t_linspace=t_linspace,
            )

        if output_df:
            X_df = pd.DataFrame(X.T, index=T, columns=state_names)
            X_df.index.name = "time"
            return X_df
        else:
            return X, T

    @staticmethod
    def of_measurements(
        sys: System,
        *,
        X: Optional[NDArray] = None,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        X_df: Optional[pd.DataFrame] = None,
        input_df: Optional[bool] = False,
        output_df: Optional[bool] = False,
        signal: Optional[Callable] = None,
        override_system_h: Union[Callable, None, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
        override_system_measurement_names: Union[List[str], bool] = False,
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

        >>> from pykal_core.system import System

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

            T = self.safeio._standardize_time_input_to_linspace(t_vector, t_span, dt)
            if X.shape[1] != len(T):
                raise ValueError(
                    f"X has {X.shape[1]} steps, but time vector has {len(T)} steps"
                )

        measurement_names = sys.safeio._resolve_override_str_list(
            override_system_measurement_names, sys.measurement_names
        )

        h = sys.safeio._resolve_override_func(
            override_system_h, sys.h, System.h_identity
        )

        u = sys.make_u_zero()

        if signal:
            u = signal

        R = sys.safeio._resolve_override_func(
            override_system_R,
            sys.R,
            sys.make_R(measurement_names, multiply_R_by_scalar=0),
        )

        Y = []

        for k, tk in enumerate(T):
            xk = X[:, k].reshape(-1, 1)
            uk = sys.safeio.smart_call(u, x=xk, t=tk)
            yk = sys.safeio.smart_call(
                h, x=xk, u=uk, t=tk, expected_shape=(len(measurement_names), 1)
            )

            R_k = sys.safeio.smart_call(
                R,
                x=xk,
                u=uk,
                t=tk,
                expected_shape=(
                    len(measurement_names),
                    len(measurement_names),
                ),
            )
            wk = np.random.multivariate_normal(
                mean=np.zeros(R_k.shape[0]), cov=R_k
            ).reshape(-1, 1)
            yk += wk

            Y.append(yk.flatten())

        if output_df:
            Y_df = pd.DataFrame(np.array(Y), index=T, columns=measurement_names)
            Y_df.index.name = "time"
            return Y_df
        else:
            return np.array(Y).T, T

    @staticmethod
    def of_signals(
        *,
        sys: System,
        signal_generator: Callable,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        output_df: Optional[bool] = False,
    ) -> Union[pd.DataFrame, tuple[NDArray, NDArray]]:
        """
        Evaluate a time-dependent signal function over a time vector and return (n, t) array.
        """
        t_linspace = sys.safeio._standardize_time_input_to_linspace(
            t_vector, t_span, dt
        )

        outputs = []
        for tk in t_linspace:
            output = sys.safeio.smart_call(signal_generator, t=tk)
            outputs.append(output)  # each output is (n, 1)

        U = np.hstack(outputs)  # shape (n, t)

        if output_df:
            Udf = pd.DataFrame(U.T, index=t_linspace)
            return Udf

        return U, t_linspace


class Jacobian:

    @classmethod
    def _matrix_jacobian_wrt_x_u_t(
        cls,
        sys: System,
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
        f_base = sys.safeio.smart_call(func, x=xk, u=uk, t=tk)
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
            f_plus = sys.safeio.smart_call(
                func, x=xk + dx, u=uk, t=tk, expected_shape=(n_output, 1)
            )
            f_minus = sys.safeio.smart_call(
                func, x=xk - dx, u=uk, t=tk, expected_shape=(n_output, 1)
            )
            Jx[:, i : i + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂u
        for j in range(n_inputs):
            du = np.zeros_like(uk)
            du[j, 0] = epsilon
            f_plus = sys.safeio.smart_call(
                func, x=xk, u=uk + du, t=tk, expected_shape=(n_output, 1)
            )
            f_minus = sys.safeio.smart_call(
                func, x=xk, u=uk - du, t=tk, expected_shape=(n_output, 1)
            )
            Ju[:, j : j + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂t
        if include_dt:
            f_plus = sys.safeio.smart_call(
                func, x=xk, u=uk, t=tk + epsilon, expected_shape=(n_output, 1)
            )
            f_minus = sys.safeio.smart_call(
                func, x=xk, u=uk, t=tk - epsilon, expected_shape=(n_output, 1)
            )
            Jt = (f_plus - f_minus) / (2 * epsilon)
            return Jx, Ju, Jt

        return Jx, Ju

    @staticmethod
    def wrt_x(
        sys: System,
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
            Jx, _ = Jacobian._matrix_jacobian_wrt_x_u_t(
                sys, func, xk, uk, tk, epsilon=epsilon
            )
            return Jx

        return jacobian_x

    @staticmethod
    def wrt_u(
        sys: System,
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
            _, Ju = Jacobian._matrix_jacobian_wrt_x_u_t(
                sys, func, xk, uk, tk, epsilon=epsilon
            )
            return Ju

        return jacobian_u

    @staticmethod
    def wrt_t(
        sys: System,
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
            _, _, Jt = Jacobian._matrix_jacobian_wrt_x_u_t(
                sys, func, xk, uk, tk, epsilon=epsilon, include_dt=True
            )
            return Jt

        return jacobian_t


class Observability:
    @classmethod
    def matrix(
        cls,
        sys: System,
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

        state_names = (
            sys.state_names
            if override_system_state_names is False
            else sys.safeio._validate_string_sequence(override_system_state_names)
        )
        measurement_names = (
            sys.measurement_names
            if override_system_measurement_names is False
            else sys.safeio._validate_string_sequence(override_system_measurement_names)
        )
        input_names = (
            sys.input_names
            if override_system_input_names is False
            else sys.safeio._validate_string_sequence(override_system_input_names)
        )

        t_linspace = sys.safeio._standardize_time_input_to_linspace(
            t_vector, t_span, dt
        )
        n_states = len(state_names)
        m_meas = len(measurement_names)
        K = len(t_linspace)

        # (K * m, n) matrix
        O = np.zeros((K * m_meas, n_states))

        for i in range(n_states):
            ei = np.zeros_like(x0, dtype=float)
            ei[i, 0] = epsilon
            x_plus = x0 + ei
            x_minus = x0 - ei

            X_plus, _ = sys.simulate_states(
                x0=x_plus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=state_names,
                override_system_input_names=input_names,
                override_system_Q=override_system_Q,
            )

            X_minus, _ = sys.simulate_states(
                x0=x_minus,
                t_vector=t_linspace,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=state_names,
                override_system_input_names=input_names,
                override_system_Q=override_system_Q,
            )

            Y_plus, _ = sys.simulate_measurements(
                X=X_plus,
                t_vector=t_linspace,
                override_system_h=override_system_h,
                override_system_u=override_system_u,
                override_system_measurement_names=measurement_names,
                override_system_input_names=input_names,
                override_system_R=override_system_R,
            )
            Y_minus, _ = sys.simulate_measurements(
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

    @classmethod
    def grammian(
        cls,
        sys: System,
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

        O = Observability.matrix(
            sys=sys,
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

    @classmethod
    def CRB(
        cls,
        sys: System,
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
        output_sr: bool = False,
    ) -> Union[pd.Series, NDArray]:
        """
        Compute the Cramér-Rao Bound (CRB) diagonal values for each state
        from the observability Gramian.

        Parameters
        ----------
        W : NDArray
            Empirical observability Gramian, shape (n, n)

        output_df : bool, optional
            If True, return the result as a pandas Series indexed by state names.

        Returns
        -------
        Union[pd.Series, NDArray]
            - If output_df is True: CRB diagonal values as a Series indexed by state names.
            - Otherwise: NumPy array of CRB diagonal values.
        """
        ridge = 1e-5  # regularization for numerical stability
        W = Observability.grammian(
            sys=sys,
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

        CRB = np.linalg.inv(W + ridge * np.eye(W.shape[0]))
        diag_vals = np.diag(CRB)

        if output_sr:
            return pd.Series(diag_vals, index=sys.state_names, name="CRB")
        else:
            return diag_vals

    @classmethod
    def of_states_from_grammian_nullspace(
        cls,
        sys: System,
        *,
        W: NDArray,
        tol: float = 1e-10,
        output_sr: bool = False,
    ) -> Union[pd.Series, NDArray]:
        """
        Compute per-state observability indices using the nullspace projection method.

        Parameters
        ----------
        W : NDArray
            Empirical observability Gramian, shape (n, n).
        tol : float
            Threshold below which eigenvalues are considered "zero" (nullspace).
        output_df : bool, optional
            If True, return results as a pandas Series with state names as index.

        Returns
        -------
        Union[pd.Series, NDArray]
            - If output_df is True: Series mapping state name → observability index ∈ [0, 1]
            - Otherwise: NumPy array of observability indices
        """
        n = W.shape[0]
        eigvals, eigvecs = np.linalg.eigh(W)

        nullspace_mask = eigvals < tol
        null_vectors = eigvecs[:, nullspace_mask]  # shape (n, r)

        scores = np.empty(n)
        for j in range(n):
            e_j = np.zeros((n, 1))
            e_j[j, 0] = 1.0
            proj = null_vectors.T @ e_j  # shape (r, 1)
            norm2 = np.sum(proj**2)
            scores[j] = float(1.0 - norm2)

        if output_sr:
            return pd.Series(scores, index=sys.state_names, name="Observability Index")
        else:
            return scores

    @classmethod
    def of_states_over_time_from_grammian_via_callable(
        cls,
        sys: System,
        *,
        x0: NDArray,
        grammian_to_scores_func: Callable,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        divide_time_into_k_windows: Optional[int] = None,
        window_length_in_points: Optional[int] = None,
        window_length_in_time: Optional[float] = None,
        overlap_points_between_windows: Optional[int] = None,
        overlap_time_between_windows: Optional[float] = None,
        output_df: bool = False,
        epsilon: float = 1e-5,
        **kwargs,
    ) -> Union[list[pd.Series], pd.DataFrame]:
        """
        Compute per-state observability scores over time windows using any method
        that maps a Gramian to a statewise score (e.g., nullspace, CRB).

        Parameters
        ----------
        x0 : NDArray
            Initial state.
        grammian_to_scores_func : Callable
            A function that accepts W (Gramian) and returns a score per state.
            Must accept `sys` and `W` as arguments.
        t_span, dt, t_vector : time config
            Either t_span + dt or explicit t_vector.
        window_* : windowing config
            Controls how time is split into windows.
        output_df : bool
            If True, return as time-indexed DataFrame.
        epsilon : float
            Perturbation magnitude for finite differences.
        kwargs : dict
            Additional arguments passed to `grammian_to_scores_func`.

        Returns
        -------
        Union[list[pd.Series], pd.DataFrame]
            List or time-indexed DataFrame of per-state scores over time.
        """
        t_linspace = sys.safeio._standardize_time_input_to_linspace(
            t_vector, t_span, dt
        )
        list_of_t_windows = sys.safeio._standardize_make_time_windows(
            t_linspace,
            divide_time_into_k_windows=divide_time_into_k_windows,
            window_length_in_points=window_length_in_points,
            window_length_in_time=window_length_in_time,
            overlap_points_between_windows=overlap_points_between_windows,
            overlap_time_between_windows=overlap_time_between_windows,
            dt=dt,
        )

        scores_over_time = []
        times = []

        for window_t in list_of_t_windows:
            W = cls.grammian(sys=sys, x0=x0, t_vector=window_t, epsilon=epsilon)
            sr = sys.safeio.smart_call(
                grammian_to_scores_func, sys=sys, W=W, output_sr=True, **kwargs
            )
            scores_over_time.append(sr)
            times.append(window_t[len(window_t) // 2])  # midpoint of window

        if output_df:
            df = pd.DataFrame(scores_over_time, index=times)
            df.index.name = "time"
            return df
        else:
            return scores_over_time

    @classmethod
    def of_states_from_P_projection_onto_canonical_basis(
        cls,
        sys: System,
        *,
        P0: NDArray,
        Pk: NDArray,
        eps: float = 1e-10,
        output_sr: bool = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
    ) -> Union[NDArray, pd.Series]:
        """
        Compute normalized state-wise uncertainty by projecting the posterior
        covariance onto canonical basis directions after whitening by P0.

        Each value reflects how much normalized uncertainty lies along a state axis.

        Returns
        -------
        NDArray or pd.Series
            Normalized posterior variances along canonical basis directions.
            Higher values indicate greater uncertainty.
        """

        try:
            L = cholesky(P0)
        except LinAlgError:
            L = cholesky(P0 + eps * np.eye(P0.shape[0]))

        P0_inv_sqrt = inv(L)
        Pk_dimensionless = P0_inv_sqrt @ Pk @ P0_inv_sqrt
        Pk_normalized = Pk_dimensionless / np.trace(Pk_dimensionless)

        statewise_variance_normalized = np.diag(Pk_normalized)

        if output_sr:
            state_names = sys.safeio._resolve_override_str_list(
                override_system_state_names, sys.state_names
            )
            return pd.Series(
                statewise_variance_normalized,
                index=state_names,
                name="Normalized Canonical Variance",
            )
        else:
            return statewise_variance_normalized

    @classmethod
    def of_states_from_P_spread_across_canonical_basis(
        cls,
        sys: System,
        *,
        P0: NDArray,
        Pk: NDArray,
        eps: float = 1e-10,
        output_sr: bool = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
    ) -> Union[NDArray, pd.Series]:
        """
        Compute the total normalized covariance spread across each canonical basis direction.

        This computes the row-sum of the trace-normalized, dimensionless covariance matrix:
            P̃ = inv(sqrt(P0)) @ Pk @ inv(sqrt(P0))
        Each row sum reflects how much total uncertainty is associated with that state,
        including its correlations with other states.

        Returns
        -------
        beta : NDArray or pd.Series
            Per-state uncertainty spread over canonical basis (higher = more correlated).
        """

        try:
            L = cholesky(P0)
        except LinAlgError:
            L = cholesky(P0 + eps * np.eye(P0.shape[0]))

        P0_inv_sqrt = inv(L)
        Pk_dimensionless = P0_inv_sqrt @ Pk @ P0_inv_sqrt
        Pk_normalized = Pk_dimensionless / np.trace(Pk_dimensionless)

        # Row sum of normalized dimensionless covariance matrix
        row_sums = np.sum(np.abs(Pk_normalized), axis=1)  # (n,) vector

        # Optional naming support
        if output_sr:
            state_names = sys.safeio._resolve_override_str_list(
                override_system_state_names, sys.state_names
            )
            return pd.Series(row_sums, index=state_names, name="Covariance Spread")
        else:
            return row_sums

    @classmethod
    def of_states_over_time_from_P_series_via_callable(
        cls,
        sys: System,
        *,
        P_series: pd.Series,
        method: Callable,
        eps: float = 1e-10,
        output_df: bool = False,
        override_system_state_names: Union[Sequence[str], bool] = False,
        **kwargs,
    ) -> Union[pd.DataFrame, list[Union[NDArray, pd.Series]]]:
        """
        Evaluate a user-supplied `of_states_from_P_*` method over all time steps
        in a time-indexed Series of covariance matrices.

        Parameters
        ----------
        P_series : pd.Series
            A time-indexed Series of (n x n) covariance matrices.
        method : Callable
            A function like `of_states_from_P_projection_onto_canonical_basis`
            that accepts P0 and Pk and returns per-state observability.
        eps : float, default 1e-10
            Regularization for inversion/Cholesky.
        output_df : bool, default False
            If True, return a time-indexed DataFrame. Otherwise, return a list.
        override_system_state_names : Sequence[str] or bool, optional
            Custom state names for Series/DataFrame output.
        kwargs : dict
            Additional keyword arguments forwarded to the method.

        Returns
        -------
        Union[pd.DataFrame, list]
            - DataFrame if `output_df=True`: rows indexed by time, columns by state.
            - Otherwise, a list of arrays or Series (one per time).
        """
        P0 = P_series.iloc[0]

        results = []
        times = []

        for tk, Pk in P_series.items():
            result = sys.safeio.smart_call(
                method,
                P0=P0,
                Pk=Pk,
                eps=eps,
                sys=sys,
                output_sr=True,
                override_system_state_names=override_system_state_names,
                **kwargs,
            )
            results.append(result)
            times.append(tk)

        if output_df:
            df = pd.DataFrame(results, index=times)
            df.index.name = "time"
            return df
        else:
            return results

    @staticmethod
    def plot_observability_directions_from_eigenpairs(
        eigvals: NDArray,
        eigvecs: NDArray,
        state_names: Optional[Sequence[str]] = None,
        projection_dims: tuple[int, int] = (0, 1),
        title: str = "Projected Observability Directions (2D)",
    ) -> None:
        """
        Plot eigenvectors of the observability Gramian projected into 2D to visualize
        most/least observable directions.

        Parameters
        ----------
        eigvals : NDArray
            1D array of eigenvalues (length n).
        eigvecs : NDArray
            2D array of eigenvectors (shape n x n), columns are eigenvectors.
        state_names : Sequence[str], optional
            Names of the states. Used to label axes. If None, index labels are used.
        projection_dims : tuple[int, int]
            Indices of the two state dimensions to project onto (e.g., (0,1)).
        title : str
            Title of the plot.
        """
        n = eigvals.shape[0]
        if eigvecs.shape != (n, n):
            raise ValueError(f"eigvecs must be of shape ({n}, {n}) to match eigvals")

        i, j = projection_dims
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(
                f"Invalid projection dimensions {projection_dims} for n={n}"
            )

        # Sort by decreasing observability
        idx = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        # Normalize eigenvalue magnitudes
        scales = eigvals_sorted / np.max(eigvals_sorted)

        fig, ax = plt.subplots(figsize=(6, 6))
        origin = np.zeros(2)

        for k in range(n):
            vec_proj = eigvecs_sorted[[i, j], k] * scales[k]
            color = "r" if k == 0 else ("b" if k == n - 1 else "gray")
            label = (
                "Most observable"
                if k == 0
                else ("Least observable" if k == n - 1 else None)
            )
            ax.quiver(
                *origin,
                *vec_proj,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                label=label,
            )

        xlabel = (
            state_names[i] if state_names and len(state_names) > i else f"State {i}"
        )
        ylabel = (
            state_names[j] if state_names and len(state_names) > j else f"State {j}"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.set_aspect("equal")
        ax.legend()
        plt.tight_layout()
        plt.show()


class Error:
    def error_pointwise(
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
        >>> from pykal_core.utils.utils_computation import Error
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

    def error_nees(
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
        >>> from pykal_core.utils.utils_computation import Error
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

    def error_nll(
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
        >>> from pykal_core.utils.utils_computation import Error
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

    def error_mse_per_state(truedf: pd.DataFrame, estdf: pd.DataFrame) -> pd.Series:
        """
        Compute mean squared error per state over all time steps.

        Returns
        -------
        pd.Series
            MSE per state (averaged over time).

        Examples
        --------
        >>> import pandas as pd
        >>> from pykal_core.utils.utils_computation import Error
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


class Matrix:
    @staticmethod
    def eigenvalues_and_eigenvectors_of_symmetric_matrix(
        M: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the eigenvalues and eigenvectors of a symmetric matrix.

        Parameters
        ----------
        M : NDArray
            A square symmetric matrix (e.g., observability Gramian)

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple (eigvals, eigvecs) where:
            - eigvals is a 1D array of eigenvalues sorted in descending order.
            - eigvecs is a matrix whose columns are the corresponding eigenvectors.
        """
        eigvals, eigvecs = np.linalg.eigh(M)  # stable for symmetric matrices
        idx = np.argsort(eigvals)[::-1]  # sort indices descending
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]
        return eigvals_sorted, eigvecs_sorted

    @staticmethod
    def condition_number_from_eigenvalues(eigvals: NDArray) -> float:
        """
        Compute the 2-norm condition number from a list of eigenvalues.

        Parameters
        ----------
        eigvals : NDArray
            A 1D array of real eigenvalues, typically from a symmetric matrix.

        Returns
        -------
        float
            Condition number (ratio of largest to smallest eigenvalue)
        """
        eigvals_sorted = np.sort(eigvals)[::-1]  # sort descending
        λ_max = eigvals_sorted[0]
        λ_min = eigvals_sorted[-1]
        if np.isclose(λ_min, 0.0):
            return np.inf
        return λ_max / λ_min
