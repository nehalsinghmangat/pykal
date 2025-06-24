from functools import wraps
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import Callable, Optional, List, Union, Sequence
from utils.utils_safeio import SafeIO as safeio
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


class SystemIO:
    """
    Interface for registering validated components of a dynamical system model.

    `SystemIO` extends `SafeIO` to support structured registration and validation of
    user-defined system functions, including dynamics (`f`), measurements (`h`),
    noise models (`Q`, `R`), and naming conventions for states, inputs, and outputs.

    All registered functions are validated with strict rules:
      • Parameter names must match known aliases (`x`, `u`, `t`, etc.)
      • All parameters and return types must be type-annotated
      • Return type must be `NDArray[...]`

    Additionally, default functions are provided when `Q`, `R`, or `u` are unspecified.

    Inherits
    --------
    SafeIO
        Provides core signature validation, alias checking, argument injection,
        and shape enforcement utilities.

    Methods
    -------
    set_f(f: Callable) -> Callable
        Register system dynamics function `f(x, u, t)` with validation.
    set_h(h: Callable) -> Callable
        Register measurement function `h(x, u, t)` with validation.
    set_u(u: Optional[Callable]) -> Callable
        Register or default control input function `u(t)`.
    set_Q(Q: Callable | None, state_names: Sequence[str]) -> Callable
        Register or default process noise covariance `Q(x, u, t)`.
    set_R(R: Callable | None, measurement_names: Sequence[str]) -> Callable
        Register or default measurement noise covariance `R(x, u, t)`.
    set_state_names(names: Sequence[str]) -> Sequence[str]
        Register list of state variable names (with order).
    set_measurement_names(names: Sequence[str]) -> Sequence[str]
        Register list of measurement variable names (with order).
    set_input_names(names: Sequence[str] | None) -> Sequence[str] | None
        Register list of input names or return `None`.
    set_system_type(system_type: SystemType | None) -> SystemType
        Register or default to a `SystemType` enum member.

    Notes
    -----
    - Signature and type validation for all function registrations is handled by
      the `@verify_signature_and_parameter_names` decorator.
    - All string-name setters (e.g., for states or measurements) are checked
      with `@verify_ordered_string_sequence` for consistency and ordering.
    - Enum validation for `SystemType` uses `@verify_system_type`.

    Examples
    --------
    >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return x + u
    >>> SystemIO.set_f(f)
    <function f at ...>

    >>> SystemIO.set_state_names(["x1", "x2"])
    ['x1', 'x2']

    >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
    ...     return np.eye(len(x))
    >>> SystemIO.set_Q(Q, state_names=["x1", "x2"])
    <function Q at ...>
    """

    @staticmethod
    def verify_system_type(func: Callable) -> Callable:
        """
        Decorator to validate that the first argument is a valid SystemType enum member.

        Raises
        ------
        TypeError
            If the first argument is not a valid SystemType.
        """

        @wraps(func)
        def wrapper(system_type, *args, **kwargs):
            if not isinstance(system_type, SystemType):
                raise TypeError(
                    f"Expected SystemType, got {type(system_type).__name__}"
                )
            return func(system_type, *args, **kwargs)

        return wrapper

    @staticmethod
    def verify_ordered_string_sequence(param_name: str) -> Callable:
        """
        Decorator to ensure the first argument is a list or tuple of strings.

        Parameters
        ----------
        param_name : str
            Name used in error messages to indicate what is being validated.

        Returns
        -------
        Callable
            A decorator that validates the first argument of the decorated function.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(names: Union[list, tuple], *args, **kwargs):
                if not isinstance(names, (list, tuple)):
                    raise TypeError(
                        f"{param_name} must be a list or tuple, got {type(names).__name__}"
                    )
                if not all(isinstance(v, str) for v in names):
                    raise TypeError(f"All elements in {param_name} must be strings")
                return func(names, *args, **kwargs)

            return wrapper

        return decorator

    @safeio.verify_signature_and_parameter_names
    @staticmethod
    def set_f(f: Optional[Callable]) -> Callable:
        """
        Set the system dynamics function after validating its signature.

        The function must return an NDArray and accept a subset of known alias names
        for state (`x`), input (`u`), and time (`t`).

        Examples
        --------
        >>> import numpy as np
        >>> from numpy.typing import NDArray



        # ✅ VALID: (x, u, t) -> NDArray
        >>> def good_f(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x + u
        >>> SystemIO.set_f(good_f)
        <function good_f...

        # ✅ VALID: no arguments, returns NDArray
        >>> def f_no_args() -> NDArray:
        ...     return np.zeros((2, 1))
        >>> SystemIO.set_f(f_no_args)
        <function f_no_args...

        # ✅ VALID: different argument order, different aliases (tk, state, uk)
        >>> def f_ordered(t: float, x: NDArray, u: NDArray) -> NDArray:
        ...     return x + u
        >>> SystemIO.set_f(f_ordered)
        <function f_ordered...

        # ✅ VALID: subset of arguments (t, x)
        >>> def f_partial(t: float, x: NDArray) -> NDArray:
        ...     return x * 2
        >>> SystemIO.set_f(f_partial)
        <function f_partial...

        # ✅ VALID: single time argument using alias `tau`
        >>> def f_timeonly(tau: float) -> NDArray:
        ...     return np.ones((3, 1))
        >>> SystemIO.set_f(f_timeonly)
        <function f_timeonly...

        # INVALID: None
        >>> SystemIO.set_f(None)
        Traceback (most recent call last):
        ...
        TypeError: The system function f cannot be None

        # ❌ UNRECOGNIZED PARAMETER NAME
        >>> def bad_f1(z: NDArray) -> NDArray:
        ...     return z
        >>> SystemIO.set_f(bad_f1)
        Traceback (most recent call last):
        ...
        TypeError: In function `bad_f1`, parameter `z` is not a recognized alias.
        Expected one of: ['input', 'state', 't', 't_k', 'tau', 'time', 'u', 'u_k', 'x', 'x_k']

        # ❌ MISSING TYPE ANNOTATION
        >>> def bad_f2(x) -> NDArray:
        ...     return x
        >>> SystemIO.set_f(bad_f2)
        Traceback (most recent call last):
        ...
        TypeError: In function `bad_f2`, input `x` is missing a type annotation.

        # ❌ WRONG TYPE FOR x
        >>> def bad_f3(x: float) -> NDArray:
        ...     return np.array([[x]])
        >>> SystemIO.set_f(bad_f3)
        Traceback (most recent call last):
        ...
        TypeError: In function `bad_f3`, input `x` must be NDArray[...] or Optional[NDArray], got <class 'float'>

        # ❌ WRONG TYPE FOR t
        >>> def bad_f4(x: NDArray, u: NDArray, t: NDArray) -> NDArray:
        ...     return x + u
        >>> SystemIO.set_f(bad_f4)
        Traceback (most recent call last):
        ...
        TypeError: In function `bad_f4`, input `t` must be float or Optional[float], got numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]

        # ❌ MISSING RETURN ANNOTATION
        >>> def bad_f5(x: NDArray):
        ...     return x
        >>> SystemIO.set_f(bad_f5)
        Traceback (most recent call last):
        ...
        TypeError: In function `bad_f5`, missing return type annotation. Expected NDArray[...]

        # ❌ RETURN TYPE IS NOT NDArray
        >>> def bad_f6(x: NDArray) -> float:
        ...     return 1.0
        >>> SystemIO.set_f(bad_f6)
        Traceback (most recent call last):
        ...
        TypeError: In function `bad_f6`, return type must be NDArray[...], got <class 'float'>

        """

        if f is None:
            raise TypeError("The system function f cannot be None")
        return f

    @safeio.verify_signature_and_parameter_names
    @staticmethod
    def set_h(h: Optional[Callable]) -> Callable:
        """
        Set the system measurement function after validating its signature.

        This method performs the same validation as `set_f`, ensuring that the
        user-supplied function `h` has allowed argument names (like x, u, t) and
        correct type annotations, and that its return type is NDArray.

        See Also
        --------
        set_f : Valid usage examples and decorator-based validation rules.
        """
        if h is None:
            raise TypeError("The measurement function h cannot be None")
        return h

    @safeio.verify_signature_and_parameter_names
    @staticmethod
    def set_u(u: Optional[Callable]) -> Callable:
        """
        Set the system input function after validating its signature.

        If `u` is None, this sets a default function that returns a zero input vector
        of shape (1, 1) for any time `t`. Otherwise, the callable is returned unchanged
        (but validated via decorator).

        Examples
        --------
        >>> u = SystemIO.set_u(None)
        >>> u(0.0)
        array([[0.]])

        See Also
        --------
        set_f : Full examples of valid and invalid signatures.
        set_h : Measurement function validation, using the same decorator.
        """
        if u is None:

            def default_u(t: NDArray) -> NDArray:
                return np.zeros((1, 1))

            return default_u
        else:
            return u

    @staticmethod
    def set_Q(Q: Union[Callable, bool, None], state_names: Sequence[str]) -> Callable:
        """
        Set the process noise covariance function `Q(x, u, t)`.

        This registers a user-defined function for computing the process noise
        covariance matrix. The function must return an `NDArray` of shape `(n, n)`,
        where `n = len(state_names)`.

        If `Q` is:
        - `None`: returns a default function that returns the zero matrix.
        - Callable: validates its signature and return type against standard system
          function rules (via `verify_signature_and_parameter_names`).

        Parameters
        ----------
        Q : Callable or None
            User-defined function of the form `Q(x, u, t) -> NDArray`, or `None`
            to default to a zero matrix.
        state_names : Sequence[str]
            List of state variable names used to determine the size of the default matrix.

        Returns
        -------
        Callable
            A validated (or default) covariance function.

        Examples
        --------
        >>> def valid_Q(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return np.eye(len(x))

        >>> Q_checked = SystemIO.set_Q(valid_Q, state_names=["x1", "x2"])
        >>> Q_checked(np.zeros((2, 1)), np.zeros((2, 1)), 0.0)
        array([[1., 0.],
               [0., 1.]])

        >>> Q_default = SystemIO.set_Q(None, state_names=["x1", "x2", "x3"])
        >>> Q_default(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

        See Also
        --------
        set_f : Registers the system dynamics model.
        set_h : Registers the measurement function.
        set_u : Registers the control input function.
        """
        if Q is None:

            def default_Q(x: NDArray, u: NDArray, t: float) -> NDArray:
                return np.zeros((len(state_names), len(state_names)))

            return default_Q
        else:

            @safeio.verify_signature_and_parameter_names
            def verify_Q(Q: Callable) -> Callable:
                return Q

            return verify_Q(Q)

    @staticmethod
    def set_R(
        R: Union[Callable, bool, None], measurement_names: Sequence[str]
    ) -> Callable:
        """
        Set the measurement noise covariance function `R(x, u, t)`.

        This registers a user-defined function that computes the measurement noise
        covariance matrix. The function must return an `NDArray` of shape `(m, m)`,
        where `m = len(measurement_names)`.

        If `R` is:
        - `None`: returns a default function that returns a zero matrix of shape (m, m)
        - Callable: validated for signature and return type using the same rules as
          `set_f`, `set_h`, and `set_Q` via `@verify_signature_and_parameter_names`.

        Parameters
        ----------
        R : Callable or None
            A user-defined function of the form `R(x, u, t) -> NDArray`, or `None` to
            return a default zero matrix.
        measurement_names : Sequence[str]
            List of measurement names used to determine matrix shape.

        Returns
        -------
        Callable
            A validated (or default) covariance function.

        Examples
        --------
        >>> import numpy as np
        >>> from numpy.typing import NDArray

        >>> def valid_R(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.1 * np.eye(len(x))

        >>> R_checked = SystemIO.set_R(valid_R, measurement_names=["y1", "y2"])
        >>> R_checked(np.zeros((2, 1)), np.zeros((2, 1)), 0.0)
        array([[0.1, 0. ],
               [0. , 0.1]])

        >>> R_default = SystemIO.set_R(None, measurement_names=["y1", "y2", "y3"])
        >>> R_default(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

        See Also
        --------
        set_f : Register the system dynamics function.
        set_h : Register the measurement function.
        set_u : Register the control input function.
        set_Q : Register the process noise covariance function.
        """

        if R is None:

            def default_R(x: NDArray, u: NDArray, t: float) -> NDArray:
                return np.zeros((len(measurement_names), len(measurement_names)))

            return default_R

        @safeio.verify_signature_and_parameter_names
        def verified_R(R_fn: Callable) -> Callable:
            return R_fn

        return verified_R(R)

    @staticmethod
    @verify_ordered_string_sequence("state_names")
    def set_state_names(
        names: Union[list[str], tuple[str, ...]],
    ) -> Union[list[str], tuple[str, ...]]:
        return names

    @staticmethod
    @verify_ordered_string_sequence("measurement_names")
    def set_measurement_names(
        names: Union[list[str], tuple[str, ...]],
    ) -> Union[list[str], tuple[str, ...]]:
        return names

    @staticmethod
    def set_input_names(
        names: Optional[Union[list[str], tuple[str, ...]]],
    ) -> Union[list[str], tuple[str, ...]]:
        if names is None:
            return ["zero_input"]

        @SystemIO.verify_ordered_string_sequence("input_names")
        def check_input_names(
            names_: Union[list[str], tuple[str, ...]],
        ) -> Union[list[str], tuple[str, ...]]:
            return names_

        return check_input_names(names)

    @staticmethod
    def set_system_type(system_type: Optional[SystemType]):
        if system_type is None:
            return SystemType.CONTINUOUS_TIME_INVARIANT
        else:

            @SystemIO.verify_system_type
            def check_system_type(system_type):
                return system_type

        return check_system_type(system_type)


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
    - Full static validation of function signatures and shapes via `SystemIO` and `safeio`
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
    safeio : Core function signature/type enforcement logic.
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

        self._f = SystemIO.set_f(f)
        self._h = SystemIO.set_h(h)
        self._u = SystemIO.set_u(u)
        self._Q = SystemIO.set_Q(Q, state_names)
        self._R = SystemIO.set_R(R, measurement_names)
        self._state_names = SystemIO.set_state_names(state_names)
        self._measurement_names = SystemIO.set_measurement_names(measurement_names)
        self._input_names = SystemIO.set_input_names(input_names)
        self._system_type = SystemIO.set_system_type(system_type)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, func):
        self._f = SystemIO.set_f(func)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, func):
        self._h = SystemIO.set_h(func)

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, func):
        self._Q = SystemIO.set_Q(func, self._state_names)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, func):
        self._R = SystemIO.set_R(func, self._measurement_names)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, func):
        self._u = SystemIO.set_u(func)

    @property
    def state_names(self):
        return self._state_names

    @state_names.setter
    def state_names(self, names):
        self._state_names = SystemIO.set_state_names(names)

    @property
    def measurement_names(self):
        return self._measurement_names

    @measurement_names.setter
    def measurement_names(self, names):
        self._measurement_names = SystemIO.set_measurement_names(names)

    @property
    def input_names(self):
        return self._input_names

    @input_names.setter
    def input_names(self, names):
        self._input_names = SystemIO.set_input_names(names)

    @property
    def system_type(self):
        return self._system_type

    @system_type.setter
    def system_type(self, val):
        self._system_type = SystemIO.set_system_type(val)

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
        via `safeio.call_validated_function_with_args`, ensuring correct shape, typing,
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
        - All function calls (`f`, `u`, `Q`) are dispatched using `safeio.call_validated_function_with_args`
          to enforce safety and suppress silent failure from malformed input functions.

        See Also
        --------
        simulate_states : High-level unified simulation method.
        SystemType : Enum that distinguishes continuous from discrete systems.
        safeio.call_validated_function_with_args : Strict dispatch utility.
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
        functions are dispatched through `safeio.call_validated_function_with_args`
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

        f = self._f if override_system_f is False else SystemIO.set_f(override_system_f)
        u = self._u if override_system_u is False else SystemIO.set_u(override_system_u)
        Q = (
            self._Q
            if override_system_Q is False
            else SystemIO.set_Q(override_system_Q, self.state_names)
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

        h = self._h if override_system_h is False else SystemIO.set_h(override_system_h)
        R = (
            self._R
            if override_system_R is False
            else SystemIO.set_R(override_system_R, self.measurement_names)
        )
        u = self._u if override_system_u is False else SystemIO.set_u(override_system_u)

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
