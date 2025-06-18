import inspect
from typing import Callable, Set, get_type_hints, get_origin
import numpy as np
from numpy.typing import NDArray


def _get_alias_for_x() -> Set[str]:
    return {"x", "x_k", "state"}


def _get_alias_for_u() -> Set[str]:
    return {"u", "u_k", "input"}


def _get_alias_for_t() -> Set[str]:
    return {"t", "t_k", "time", "tau"}


def check_if_function_output_NDArray(func: Callable) -> Callable:
    """
    Ensures that the given function is annotated to return an NDArray.

    Parameters
    ----------
    func : Callable
        A function that should return an `NDArray`.

    Returns
    -------
    func : Callable
        The original function if its return annotation is correct.

    Raises
    ------
    TypeError
        If the return annotation is missing or is not a subtype of NDArray.

    Examples
    --------
    >>> def good_fun() -> NDArray[np.float64]:
    ...     return np.array([0.0])
    >>> check_if_function_output_NDArray(good_fun)  # Returns the function
    <function good_fun at ...>

    >>> def bad_fun():
    ...     return np.array([0.0])
    >>> check_if_function_output_NDArray(bad_fun)
    Traceback (most recent call last):
        ...
    TypeError: Function `bad_fun` does not have a return type annotation; please add -> NDArray

    >>> def also_bad_fun() -> list:
    ...     return [1, 2, 3]
    >>> check_if_function_output_NDArray(also_bad_fun)
    Traceback (most recent call last):
        ...
    TypeError: Function `also_bad_fun` does not return a `NDArray`, got <class 'list'>
    """
    hints = get_type_hints(func)

    if "return" not in hints:
        raise TypeError(
            f"Function `{func.__name__}` does not have a return type annotation; please add -> NDArray"
        )

    return_type = hints["return"]
    origin = get_origin(return_type)

    if origin is not np.ndarray:
        raise TypeError(
            f"Function `{func.__name__}` does not return a `NDArray`, got {return_type}"
        )

    return func


def check_if_func_param_types(func: Callable) -> Callable:
    """
    Ensures that parameters matching standard aliases for state, input, and time
    are annotated as expected:
      - state/input aliases → NDArray[...]
      - time aliases → float

    Raises
    ------
    TypeError
        If any declared alias is missing a type annotation or has an incorrect type.

    Examples
    --------
    >>> def good(x: NDArray[np.float64], u: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    ...     return x + u
    >>> check_if_func_param_types(good)
    <function good at ...>

    >>> def bad_time(x: NDArray[np.float64], u: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    ...     return x + u
    >>> check_if_func_param_types(bad_time)
    Traceback (most recent call last):
        ...
    TypeError: Time-like input `t` must be annotated as float

    >>> def missing_annot(x, u, t):
    ...     return x + u
    >>> check_if_func_param_types(missing_annot)
    Traceback (most recent call last):
        ...
    TypeError: Input `x` is missing a type annotation
    """

    hints = get_type_hints(func)
    sig = inspect.signature(func)
    all_aliases = _get_alias_for_x() | _get_alias_for_u() | _get_alias_for_t()
    t_aliases = _get_alias_for_t()
    for name, param in sig.parameters.items():
        if name not in all_aliases:
            continue

        if name not in hints:
            raise TypeError(f"Input `{name}` is missing a type annotation")

        actual = hints[name]

        if name in t_aliases:
            if actual is not float:
                raise TypeError(f"Time-like input `{name}` must be annotated as float")
        else:  # state or input
            origin = get_origin(actual)
            if origin is not np.ndarray:
                raise TypeError(f"Input `{name}` must be annotated as NDArray[...]")

    return func


def validate_system_function(func: Callable) -> Callable:
    """
    Decorator that validates a system function using:
      - check_if_func_param_types: ensures input annotations match aliases
      - check_if_function_output_NDArray: ensures return type is NDArray[...]

    Applies static validation at definition-time.

    Examples
    --------
    >>> from numpy.typing import NDArray
    >>> import numpy as np

    >>> @validate_system_function
    ... def good(x: NDArray[np.float64], u: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    ...     return x + u
    >>> good(np.array([[1.0]]), np.array([[2.0]]), 0.0).shape
    (1, 1)

    >>> @validate_system_function
    ... def bad_output(x: NDArray[np.float64], u: NDArray[np.float64], t: float) -> list:
    ...     return [1, 2, 3]
    Traceback (most recent call last):
        ...
    TypeError: Function `bad_output` does not return a `NDArray`, got <class 'list'>

    >>> @validate_system_function
    ... def bad_input(x, u, t: float) -> NDArray[np.float64]:
    ...     return np.array([[1.0]])
    Traceback (most recent call last):
        ...
    TypeError: Input `x` is missing a type annotation

    >>> @validate_system_function
    ... def bad_time(x: NDArray[np.float64], u: NDArray[np.float64], time: NDArray[np.float64]) -> NDArray[np.float64]:
    ...     return x + u
    Traceback (most recent call last):
        ...
    TypeError: Time-like input `time` must be annotated as float
    """
    func = check_if_func_param_types(func)
    func = check_if_function_output_NDArray(func)
    return func


def call_validated_function_with_args(func: Callable, *, x=None, u=None, t=None):
    """
    Call a user-supplied function by inspecting its declared parameters and dispatching
    the correct arguments based on recognized aliases for state, input, and time.

    This function ensures:
      - Only supported parameters (from a predefined alias set) are allowed.
      - Only the parameters explicitly declared by the function are passed.
      - If no recognized parameters are present, or unsupported ones are declared, an error is raised.

    Alias sets are defined by:
      - `_get_alias_for_x()` → state aliases: {"x", "x_k", "state"}
      - `_get_alias_for_u()` → input aliases: {"u", "u_k", "input"}
      - `_get_alias_for_t()` → time aliases: {"t", "t_k", "time", "tau"}

    Parameters
    ----------
    func : Callable
        A function whose parameter names may include any subset of the supported aliases.
    x : Any, optional
        The value to pass to a state-like argument (if declared).
    u : Any, optional
        The value to pass to an input-like argument (if declared).
    t : Any, optional
        The value to pass to a time-like argument (if declared).

    Returns
    -------
    Any
        The return value of `func` when called with the appropriate subset of arguments.

    Raises
    ------
    ValueError
        If `func` declares no supported parameters, or if it declares any unsupported ones.

    Examples
    --------
    # Function with a state alias
    >>> def f1(x):
    ...     return x + 1
    >>> call_validated_function_with_args(f1, x=10)
    11

    # Uses aliases for input and state
    >>> def f2(u_k, state):
    ...     return (u_k, state)
    >>> call_validated_function_with_args(f2, x=3, u=5)
    (5, 3)

    # Uses aliases for input and time
    >>> def f3(time, input):
    ...     return time * input
    >>> call_validated_function_with_args(f3, u=4, t=6)
    24

    >>> def f_no_arg():
    ...     return None
    >>> call_validated_function_with_args(f_no_arg)

    # Invalid: unsupported parameter declared
    >>> def f_invalid2(x, y):
    ...     return None
    >>> call_validated_function_with_args(f_invalid2, x=1)
    Traceback (most recent call last):
    ...
    ValueError: Function 'f_invalid2' declares unsupported parameters: y
    """

    params = inspect.signature(func).parameters
    sig_params = list(params.keys())

    # define supported aliases
    all_aliases = _get_alias_for_x() | _get_alias_for_u() | _get_alias_for_t()

    # check for any unsupported parameters
    unsupported = [p for p in sig_params if p not in all_aliases]
    if unsupported:
        raise ValueError(
            f"Function '{func.__name__}' declares unsupported parameters: "
            f"{', '.join(unsupported)}"
        )

    # build kwargs only for supported names
    kwargs = {}
    for alias in _get_alias_for_x():
        if alias in params:
            kwargs[alias] = x
    for alias in _get_alias_for_u():
        if alias in params:
            kwargs[alias] = u
    for alias in _get_alias_for_t():
        if alias in params:
            kwargs[alias] = t
            break

    return func(**kwargs)
