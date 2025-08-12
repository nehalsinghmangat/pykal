import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union, Optional, List
from scipy.integrate import solve_ivp
from pykal_core.control_system import System
from pykal_core.utils.system import SafeIO


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
        **kwargs,
    ) -> tuple[NDArray, NDArray]:

        def wrapped_f(t_k: float, x_flat: NDArray) -> NDArray:
            x_k = x_flat.reshape(-1, 1)
            u_k = sys.safeio.smart_call(u, x=x_k, t=t_k)
            dx = sys.safeio.smart_call(
                f, x=x_k, u=u_k, t=t_k, expected_shape=(len(state_names), 1), **kwargs
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
            override_system_Q,
            sys.Q,
            sys.make_Q(state_names=state_names, multiply_eye_by_scalar=0),
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
            sys.make_R(measurement_names=measurement_names, multiply_eye_by_scalar=0),
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
    def of_observer_estimates():
        """x_est = x0
        P_est = P0

        for k in range(len(T)):
            x_est, P_est = L(
                yk=Y[:, [k]],
                uk=U[:, [k]],
                f=f,
                F=F,
                Q=Q,
                h=h,
                H=H,
                R=R,
                xk=x_est,
                Pk=P_est,
                tk=T[k],
                dt=dt,
            )
        """

    @staticmethod
    def of_controller_inputs():
        pass

    @staticmethod
    def of_signals(
        *,
        signal_generator: Callable,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        output_df: Optional[bool] = False,
    ) -> Union[pd.DataFrame, tuple[NDArray, NDArray]]:
        """
        Evaluate a time-dependent signal function over a time vector and return (n, t) array.
        """
        t_linspace = SafeIO._standardize_time_input_to_linspace(t_vector, t_span, dt)

        outputs = []
        for tk in t_linspace:
            output = signal_generator(tk)
            outputs.append(output)  # each output is (n, 1)

        U = np.hstack(outputs)  # shape (n, t)

        if output_df:
            Udf = pd.DataFrame(U.T, index=t_linspace)
            return Udf

        return U, t_linspace
