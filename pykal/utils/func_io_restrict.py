import inspect
from typing import Callable,get_type_hints
import numpy as np

def function_returns_npndarray(func: Callable) -> bool:
    """
    Returns True if `func` is explicitly annotated to return `np.ndarray`.
    Raises TypeError if return annotation is missing or incorrect.

    Examples
    --------
    >>> def good_fun() -> np.ndarray:
    ...     return np.array([0])
    >>> function_returns_npndarray(good_fun)  # Correct: returns np.ndarray and is annotated
    True

    >>> def bad_fun():
    ...     return np.array([0])
    >>> function_returns_npndarray(bad_fun)  # Missing return type annotation
    Traceback (most recent call last):
        ...
    TypeError: Function `bad_fun` does not have a return type annotation; please add -> np.ndarray

    >>> def also_bad_fun() -> list:
    ...     return [1, 2, 3]
    >>> function_returns_npndarray(also_bad_fun)  # Incorrect annotation: returns list instead of np.ndarray
    Traceback (most recent call last):
        ...
    TypeError: Function `also_bad_fun` does not return a `np.ndarray`, got <class 'list'>
    
    """
    # Extract the type hints (including return annotation)
    hints = get_type_hints(func)
    
    if 'return' not in hints:
        raise TypeError(f"Function `{func.__name__}` does not have a return type annotation; please add -> np.ndarray")
    
    if hints['return'] is not np.ndarray:
        raise TypeError(f"Function `{func.__name__}` does not return a `np.ndarray`, got {hints['return']}")
    
    return True

def call_func_with_args(func: Callable, *, x=None, u=None, t=None):
    """
    Smart wrapper that calls a user-supplied function with exactly the keyword
    arguments it declares—and fails if the signature is empty or contains
    unsupported parameters.

    Supported parameter names are:
      - for state:     `x`, `x_k`, `state`
      - for input:     `u`, `u_k`, `input`
      - for time:      `t`, `t_k`, `time`, `tau`

    Parameters
    ----------
    func : Callable
        A Python callable whose signature may include any subset of the
        supported names above.
    x : any, optional
        Value to pass to `func` if it declares one of `x`, `x_k`, or `state`.
    u : any, optional
        Value to pass to `func` if it declares one of `u`, `u_k`, or `input`.
    t : any, optional
        Value to pass to `func` if it declares one of `t`, `t_k`, `time`, or `tau`.

    Returns
    -------
    Any
        The return value from `func`.

    Raises
    ------
    ValueError
        If `func` declares none of the supported parameters, or if it declares
        any parameters outside the supported set.

    Examples
    --------
    # Function with one supported param (`x`)
    >>> def f1(x):
    ...     return x + 1
    >>> call(f1, x=10) # x is passed correctly as 10
    11  

    # Function using aliases `u_k` and `state`, both mapped from u=5 and x=3
    >>> def f2(u_k, state):
    ...     return (u_k, state)
    >>> call(f2, x=3, u=5)  # correct dispatch to aliases
    (5, 3) 

    # Function using `time` and `input`, both supported aliases
    >>> def f3(time, input):
    ...     return time * input
    >>> call(f3, x=2, u=4, t=6)  # t=6 is mapped to time, u=4 to input
    24

    # Invalid: function takes no parameters
    >>> def f_invalid(): 
    ...     return None
    >>> call(f_invalid)
    Traceback (most recent call last):
    ...
    ValueError: Function 'f_invalid' declares no supported parameters; expected one of: input, state, t, t_k, tau, time, u, u_k, x, x_k

    # Invalid: `y` is not a recognized parameter name
    >>> def f_invalid2(x, y):
    ...     return None
    >>> call(f_invalid2, x=1)
    Traceback (most recent call last):
    ...
    ValueError: Function 'f_invalid2' declares unsupported parameters: y
    """
    params = inspect.signature(func).parameters
    sig_params = list(params.keys())

    # define supported aliases
    alias_x = ('x', 'x_k', 'state')
    alias_u = ('u', 'u_k', 'input')
    alias_t = ('t', 't_k', 'time', 'tau')
    all_aliases = set(alias_x + alias_u + alias_t)

    # check for supported parameters
    provided = [p for p in sig_params if p in all_aliases]
    if not provided:
        raise ValueError(
            f"Function '{func.__name__}' declares no supported parameters; "
            f"expected one of: {', '.join(sorted(all_aliases))}"
        )

    # check for any unsupported parameters
    unsupported = [p for p in sig_params if p not in all_aliases]
    if unsupported:
        raise ValueError(
            f"Function '{func.__name__}' declares unsupported parameters: "
            f"{', '.join(unsupported)}"
        )

    # build kwargs only for supported names
    kwargs = {}
    for alias in alias_x:
        if alias in params:
            kwargs[alias] = x
    for alias in alias_u:
        if alias in params:
            kwargs[alias] = u
    for alias in alias_t:
        if alias in params:
            kwargs[alias] = t
            break

    return func(**kwargs)
