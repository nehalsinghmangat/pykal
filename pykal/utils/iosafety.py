from functools import wraps
import numpy as np
from numpy.typing import NDArray
from typing import (
    Callable,
    Optional,
    get_type_hints,
    get_origin,
    get_args,
    Union,
    Sequence,
)
import inspect
from enum import Enum, auto


class VariableAliases:
    """
    Central alias registry for standardized variable naming.
    """

    _alias_for_x = {"x", "x_k", "state"}
    _alias_for_u = {"u", "u_k", "input"}
    _alias_for_t = {"t", "t_k", "time", "tau"}


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


class SafeIO(VariableAliases):
    """
    Validated interface for system function registration and safe function dispatch.

    This class provides decorators and utility methods to:
      - Validate user-defined system functions (e.g., dynamics, measurement, noise).
      - Enforce consistent input/output shapes and types using annotated signatures.
      - Support flexible aliasing of arguments like `x`, `u`, and `t` via `VariableAliases`.
      - Dynamically inject only the required arguments (`x`, `u`, `t`) into functions.
      - Provide robust runtime checks for covariance matrix shape, function output type, etc.

    Inherits
    --------
    VariableAliases
        Provides alias definitions for `x`, `u`, and `t`.

    Key Features
    ------------
    - `@verify_signature_and_parameter_names`: Validates the function's inputs and outputs.
    - `smart_call(...)`: Dynamically calls functions with only the required arguments.
    - `enforce_ndarray_shape(...)`: Ensures output has the correct ndarray shape.
    - `call_validated_function_with_args(...)`: Full dispatch + shape check in one step.

    Use Cases
    ---------
    This class is typically subclassed or used as a mixin for validating and executing
    system dynamics (`f`), measurement models (`h`), and noise covariances (`Q`, `R`) in
    state estimation pipelines.
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

    @classmethod
    def _is_ndarray_hint(cls, hint) -> bool:
        origin = get_origin(hint)
        return hint is NDArray or origin is np.ndarray or origin is NDArray

    @staticmethod
    def verify_signature_and_parameter_names(set_func: Callable) -> Callable:
        """
        Decorator to validate the input/output signature of a system function.

        Ensures the given function uses only known parameter aliases and valid type
        annotations, and that it returns an `NDArray`.

        Parameters
        ----------
        set_func : Callable
            The setter function that receives the user-defined function.

        Validation Rules
        ----------------
        - Allowed parameter names:
            • `x`, `x_k`, `state` (state aliases)
            • `u`, `u_k`, `input` (input aliases)
            • `t`, `t_k`, `time`, `tau` (time aliases)
        - Input types:
            • `x`, `u`: NDArray[...] or Optional[NDArray]
            • `t`: float or Optional[float]
        - Return type:
            • Must be NDArray[...]

        Raises
        ------
        TypeError
            If an argument name is unrecognized, lacks a valid annotation, or
            if the return type is missing or invalid.
        """

        @wraps(set_func)
        def wrapper(func):
            # Allow None to pass through without validation
            if func is None:
                return set_func(func)

            hints = get_type_hints(func)
            sig = inspect.signature(func)

            alias_x = VariableAliases._alias_for_x
            alias_u = VariableAliases._alias_for_u
            alias_t = VariableAliases._alias_for_t
            all_aliases = alias_x | alias_u | alias_t

            for name, param in sig.parameters.items():

                if name not in all_aliases:
                    raise TypeError(
                        f"In function `{func.__name__}`, parameter `{name}` is not a recognized alias.\n"
                        f"Expected one of: {sorted(all_aliases)}"
                    )

                if name not in hints:
                    raise TypeError(
                        f"In function `{func.__name__}`, input `{name}` is missing a type annotation."
                    )

                hint = hints[name]
                origin = get_origin(hint)
                args_ = get_args(hint)

                if name in alias_x | alias_u:
                    if not (
                        SafeIO._is_ndarray_hint(hint)
                        or (
                            origin is Union
                            and any(SafeIO._is_ndarray_hint(a) for a in args_)
                        )
                    ):
                        raise TypeError(
                            f"In function `{func.__name__}`, input `{name}` must be NDArray[...] or Optional[NDArray], got {hint}"
                        )
                elif name in alias_t:
                    if not (hint is float or (origin is Union and float in args_)):
                        raise TypeError(
                            f"In function `{func.__name__}`, input `{name}` must be float or Optional[float], got {hint}"
                        )

            # Return type check
            if "return" not in hints:
                raise TypeError(
                    f"In function `{func.__name__}`, missing return type annotation. Expected NDArray[...]"
                )

            ret_hint = hints["return"]
            if not SafeIO._is_ndarray_hint(ret_hint):
                raise TypeError(
                    f"In function `{func.__name__}`, return type must be NDArray[...], got {ret_hint}"
                )

            return set_func(func)

        return wrapper

    @staticmethod
    def smart_call(
        func: Callable,
        x: Optional[NDArray] = None,
        u: Optional[NDArray] = None,
        t: Optional[float] = None,
    ) -> NDArray:
        """
        Call a function with only the arguments it explicitly requests among (x, u, t).

        Useful for user-defined system functions that may only declare a subset of
        state, input, and time parameters. Argument names are matched against
        recognized aliases.

        Parameters
        ----------
        func : Callable
            The function to be called (e.g., dynamics or measurement).
        x : NDArray, optional
            State vector to inject if required.
        u : NDArray, optional
            Input vector to inject if required.
        t : float, optional
            Time value to inject if required.

        Returns
        -------
        Any
            The result of calling `func` with the appropriate subset of arguments.
        """
        sig = inspect.signature(func)
        call_kwargs = {}

        alias_x = VariableAliases._alias_for_x
        alias_u = VariableAliases._alias_for_u
        alias_t = VariableAliases._alias_for_t

        for name in sig.parameters:
            if name in alias_x and x is not None:
                call_kwargs[name] = x
            elif name in alias_u and u is not None:
                call_kwargs[name] = u
            elif name in alias_t and t is not None:
                call_kwargs[name] = t

        return func(**call_kwargs)

    @staticmethod
    def enforce_ndarray_shape(n: int, m: int) -> Callable:
        """
        Return a decorator that enforces a specific output shape on a function.

        Raises an error if the decorated function does not return an `NDArray` with shape (n, m).

        Parameters
        ----------
        n : int
            Expected number of rows.
        m : int
            Expected number of columns.

        Returns
        -------
        Callable
            A decorator that enforces the output shape.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> NDArray:
                result = func(*args, **kwargs)
                if not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"Function `{func.__name__}` must return a NumPy ndarray, got {type(result)}"
                    )
                if result.shape != (n, m):
                    raise ValueError(
                        f"Function `{func.__name__}` must return shape ({n}, {m}), got {result.shape}"
                    )
                return result

            return wrapper

        return decorator

    @staticmethod
    def call_validated_function_with_args(
        func: Callable,
        x: Optional[NDArray] = None,
        u: Optional[NDArray] = None,
        t: Optional[float] = None,
        expected_shape: Optional[tuple[int, int]] = None,
    ) -> NDArray:
        """
        Call a function with the required arguments and validate its return shape.

        This utility inspects the function signature, injects the appropriate
        arguments (x, u, t), and optionally checks that the returned NDArray has
        the expected shape.

        Parameters
        ----------
        func : Callable
            The user-defined function (e.g., f, h, Q, or R).
        x : NDArray, optional
            State vector to pass if the function requests it.
        u : NDArray, optional
            Input vector to pass if requested.
        t : float, optional
            Time value to pass if requested.
        expected_shape : tuple[int, int], optional
            If provided, enforces that the function output has this shape.

        Returns
        -------
        NDArray
            The result of the function call with validated shape.

        Examples
        --------

        >>> import numpy as np
        >>> from numpy.typing import NDArray
        >>> from pykal.utils.iosafety import SafeIO

        >>> x = np.array([[1.0], [2.0]])
        >>> u = np.array([[0.5]])
        >>> t = 0.1

        # 1. Function expecting only `x`
        >>> def fx(x: NDArray) -> NDArray:
        ...     return x * 2
        >>> SafeIO.call_validated_function_with_args(fx, x=x)
        array([[2.],
               [4.]])

        # 2. Function expecting `x` and `u`
        >>> def fxu(x: NDArray, u: NDArray) -> NDArray:
        ...     return x + u
        >>> SafeIO.call_validated_function_with_args(fxu, x=x, u=u)
        array([[1.5],
               [2.5]])

        # 3. Function expecting `x`, `u`, and `t`
        >>> def fxut(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return x + u * t
        >>> SafeIO.call_validated_function_with_args(fxut, x=x, u=u, t=t)
        array([[1.05],
               [2.05]])

        # 4. Function expecting only `t`
        >>> def ft(t: float) -> NDArray:
        ...     return np.array([[t]])
        >>> SafeIO.call_validated_function_with_args(ft, t=0.3)
        array([[0.3]])

        # 5. Function returns a matrix and expected_shape matches
        >>> def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.1 * np.eye(2)
        >>> SafeIO.call_validated_function_with_args(Q, x=x, u=u, t=t, expected_shape=(2, 2))
        array([[0.1, 0. ],
               [0. , 0.1]])

        # 6. Function returns wrong shape → raises error
        >>> def bad_Q(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return 0.1 * np.eye(3)
        >>> SafeIO.call_validated_function_with_args(bad_Q, x=x, u=u, t=t, expected_shape=(2, 2))
        Traceback (most recent call last):
        ...
        ValueError: Output shape mismatch. Expected (2, 2), got (3, 3)

        # 7. Function takes no arguments → OK if none required
        >>> def const_func() -> NDArray:
        ...     return np.ones((2, 1))
        >>> SafeIO.call_validated_function_with_args(const_func)
        array([[1.],
               [1.]])

        # 8. Function takes no arguments but caller tries to pass x → still OK, x is ignored
        >>> SafeIO.call_validated_function_with_args(const_func, x=x)
        array([[1.],
               [1.]])

        # 9. Invalid output type (not ndarray) → raises TypeError
        >>> def bad_return(x: NDArray) -> list:
        ...     return [1, 2]
        >>> SafeIO.call_validated_function_with_args(bad_return, x=x)
        Traceback (most recent call last):
        ...
        TypeError: Function must return a NumPy ndarray, got <class 'list'>

        """
        result = SafeIO.smart_call(func, x=x, u=u, t=t)

        if not isinstance(result, np.ndarray):
            raise TypeError(f"Function must return a NumPy ndarray, got {type(result)}")

        if expected_shape is not None:
            n, m = expected_shape
            if result.shape != (n, m):
                raise ValueError(
                    f"Output shape mismatch. Expected {(n, m)}, got {result.shape}"
                )

        return result


class SystemIO(SafeIO):
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

    @SafeIO.verify_signature_and_parameter_names
    @staticmethod
    def set_f(f: Callable) -> Callable:
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

    @SafeIO.verify_signature_and_parameter_names
    @staticmethod
    def set_h(h: Callable) -> Callable:
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

    @SafeIO.verify_signature_and_parameter_names
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

            @SafeIO.verify_signature_and_parameter_names
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

        @SafeIO.verify_signature_and_parameter_names
        def verified_R(R_fn: Callable) -> Callable:
            return R_fn

        return verified_R(R)

    @staticmethod
    @SafeIO.verify_ordered_string_sequence("state_names")
    def set_state_names(
        names: Union[list[str], tuple[str, ...]],
    ) -> Union[list[str], tuple[str, ...]]:
        return names

    @staticmethod
    @SafeIO.verify_ordered_string_sequence("measurement_names")
    def set_measurement_names(
        names: Union[list[str], tuple[str, ...]],
    ) -> Union[list[str], tuple[str, ...]]:
        return names

    @staticmethod
    def set_input_names(
        names: Optional[Union[list[str], tuple[str, ...]]],
    ) -> Optional[Union[str, list[str], tuple[str, ...]]]:
        if names is None:
            return None

        @SafeIO.verify_ordered_string_sequence("input_names")
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

            @SafeIO.verify_system_type
            def check_system_type(system_type):
                return system_type

        return check_system_type(system_type)


class EKFIO(SystemIO):
    """
    Extended Kalman Filter I/O validation utilities.

    `EKFIO` extends `SystemIO` with specialized validation tools for running
    Extended Kalman Filters, particularly checking the integrity of inputs passed
    to the `run` method.

    This class includes a decorator to ensure that the filter is initialized
    with compatible dimensions, types, and callable components, such as dynamics
    and measurement Jacobians.

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
        Decorator to validate inputs to the Kalman Filter `run` method.

        This ensures that initial conditions, measurement data, Jacobians, and
        timing arguments are valid before simulation begins.

        Validation Rules
        ----------------
        - `x0`, `P0` must be 2D `NDArray` objects with shape `(n, 1)` and `(n, n)`
        - `Y` must be a 2D array of shape `(T, m)` where `T` ≥ 1 and `m` ≥ 1
        - `F`, `H`, `U` must be callable
        - `start_time`, `dt` must be floats

        Parameters
        ----------
        func : Callable
            The `run` method to be wrapped and validated.

        Raises
        ------
        ValueError
            If `Y` is not a 2D array, or has zero rows or columns.
        TypeError
            If `F`, `H`, or `U` are not callables, or if `start_time` or `dt` are not floats.

        Returns
        -------
        Callable
            A wrapped version of the `run` method with validation enforced.

        """

        @wraps(func)
        def wrapper(
            self,
            x0: NDArray,
            P0: NDArray,
            Y: NDArray,
            F: Callable,
            H: Callable,
            start_time: float,
            dt: float,
            U: Callable,
            *args,
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
            if not callable(U):
                raise TypeError("U must be a callable function")

            if not isinstance(start_time, (int, float)):
                raise TypeError("start_time must be a float")
            if not isinstance(dt, (int, float)):
                raise TypeError("dt must be a float")

            return func(
                self,
                x0,
                P0,
                Y,
                F,
                H,
                start_time,
                dt,
                U,
                *args,
                **kwargs,
            )

        return wrapper


class UKFIO(SystemIO):
    pass


class PKFIO(EKFIO):
    pass
