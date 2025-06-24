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
)
import inspect 

class VariableAliases:
    """
    Central alias registry for standardized variable naming.
    """

    _alias_for_x = {"x", "x_k", "state"}
    _alias_for_u = {"u", "u_k", "input"}
    _alias_for_t = {"t", "t_k", "time", "tau"}

    
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





