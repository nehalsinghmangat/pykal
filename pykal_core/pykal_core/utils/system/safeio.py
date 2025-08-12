import numpy as np
from numpy.typing import NDArray
import inspect
from typing import (
    Any,
    Callable,
    Optional,
    get_type_hints,
    get_origin,
    Sequence,
    get_args,
    Union,
    List,
    Tuple,
)
class SafeIO:
    """
    Validated interface for system function registration and safe function dispatch.
    """

    def __init__(self, parent: "System"):
        self.parent = parent
        self._aliases_for_x = ["x", "x_k", "xk", "state"]
        self._aliases_for_u = ["u", "u_k", "uk", "input"]
        self._aliases_for_t = ["t", "t_k", "tk", "time", "tau"]
        self.smart_call = self.smart_call

    @property
    def aliases_for_x(self):
        return self._aliases_for_x

    @property
    def aliases_for_u(self):
        return self._aliases_for_u

    @property
    def aliases_for_t(self):
        return self._aliases_for_t

    @aliases_for_x.setter
    def aliases_for_x(self, list_of_str):
        self._system_type = self._validate_string_sequence(list_of_str)

    @aliases_for_u.setter
    def aliases_for_u(self, list_of_str):
        self._system_type = self._validate_string_sequence(list_of_str)

    @aliases_for_t.setter
    def aliases_for_t(self, list_of_str):
        self._system_type = self._validate_string_sequence(list_of_str)

    @classmethod
    def _validate_string_sequence(cls, names: Any) -> List:
        """
        Validate that `names` is a list, tuple, or set of strings.

        Parameters
        ----------
        param_name : str
            The name of the parameter (used for error messages).
        names : Any
            The object to validate.

        Returns
        -------
        Any
            The validated `names` object, unchanged.

        Raises
        ------
        TypeError
            If `names` is not a list/tuple/set of strings.
        """

        if names is None:
            return []

        if not isinstance(names, list):
            raise TypeError(f"{names} must be a list!")

        if not all(isinstance(v, str) for v in names):
            raise TypeError(f"All elements in {names} must be strings")
        return names

    @classmethod
    def _is_ndarray_hint(cls, hint: Any) -> bool:
        """
        Check whether a type hint corresponds to or contains a NumPy ndarray.

        Handles direct, optional, and union-wrapped ndarray types.

        Examples:
        - NDArray ✔
        - np.ndarray ✔
        - Optional[NDArray] ✔
        - Union[np.ndarray, None] ✔
        """

        def unwrap_union(h: Any) -> list:
            origin = get_origin(h)
            if origin is Union:
                return [t for arg in get_args(h) for t in unwrap_union(arg)]
            return [h]

        for subtype in unwrap_union(hint):
            origin = get_origin(subtype)
            if subtype is NDArray or origin is NDArray:
                return True
            if subtype is np.ndarray or origin is np.ndarray:
                return True
        return False

    @classmethod
    def _is_tuple_of_ndarrays_hint(cls, hint: Any) -> bool:
        """
        Check whether a type hint is a Tuple of NDArray/np.ndarray types,
        including wrapped in Optional or Union.

        Examples:
        - Tuple[NDArray, NDArray] ✔
        - Optional[Tuple[np.ndarray, NDArray]] ✔
        - Union[Tuple[NDArray, NDArray], None] ✔
        """
        origin = get_origin(hint)
        if origin is Union:
            return any(
                cls._is_tuple_of_ndarrays_hint(arg) for arg in get_args(hint)
            )

        if origin not in (tuple, Tuple):
            return False

        args = get_args(hint)
        if not args:
            return False

        for arg in args:
            if not cls._is_ndarray_hint(arg):
                return False

        return True

    @classmethod
    def _is_optional_ndarray_hint(cls, hint: Any) -> bool:
        """Check if hint is Optional[NDArray] or Union[NDArray, None]."""
        origin = get_origin(hint)
        args = get_args(hint)

        return (
            origin is Union
            and any(cls._is_ndarray_hint(arg) for arg in args)
            and type(None) in args
        )

    @classmethod
    def _is_float_hint(cls, hint: Any) -> bool:
        """
        Returns True if `hint` is a float or Union containing float.

        Parameters
        ----------
        hint : Any
            A type annotation to check.

        Returns
        -------
        bool
            True if `hint` is float-compatible.
        """
        origin = get_origin(hint)
        args_ = get_args(hint)

        return (
            hint is float
            or origin is float
            or (origin is Union and any(a is float for a in args_))
        )

    @classmethod
    def _validate_func_ndarray_output_shape(
        cls, func: Callable, result: Any, n: int, m: int
    ) -> NDArray:
        """
        Enforce that the result of a function is an NDArray with shape (n, m).

        Parameters
        ----------
        func : Callable
            The function whose output is being validated.
        result : Any
            The output to validate.
        n : int
            Expected number of rows.
        m : int
            Expected number of columns.

        Returns
        -------
        NDArray
            The validated result.

        Raises
        ------
        TypeError
            If the result is not a NumPy ndarray.
        ValueError
            If the result does not have shape (n, m).
        """
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"Function `{func.__name__}` must return a NumPy ndarray, got {type(result)}"
            )
        if result.shape != (n, m):
            raise ValueError(
                f"Function `{func.__name__}` must return shape ({n}, {m}), got {result.shape}"
            )
        return result

    def _validate_func_signature(self, func: Callable) -> Callable:
        """
        Validates the input/output signature of a user-defined function
        using aliases from the given instance.

        Parameters
        ----------
        func : Callable
            The function to validate.

        Raises
        ------
        TypeError
            If required parameter names are unrecognized, type annotations are missing,
            or if the return type is missing or invalid.
        """
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        alias_x = set(self._aliases_for_x)
        alias_u = set(self._aliases_for_u)
        alias_t = set(self._aliases_for_t)

        expected_aliases = alias_x | alias_u | alias_t

        for name, param in sig.parameters.items():
            if name not in expected_aliases:
                # Allow extra parameters if they are keyword-only or have defaults
                if param.kind not in {
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }:
                    raise TypeError(
                        f"In `{func.__name__}`, unexpected parameter `{name}` must be keyword-only "
                        f"or have a default to be accepted."
                    )

            if name not in hints:
                raise TypeError(
                    f"In `{func.__name__}`, parameter `{name}` is missing a type annotation."
                )

            hint = hints[name]

            if name in alias_x:
                if not self._is_ndarray_hint(hint):
                    raise TypeError(
                        f"In `{func.__name__}`, parameter `{name}` must be NDArray[...] "
                        f"but got {hint}"
                    )

            if name in alias_u:
                if not (
                    self._is_ndarray_hint(hint)
                    or self._is_optional_ndarray_hint(hint)
                ):
                    raise TypeError(
                        f"In `{func.__name__}`, parameter `{name}` must be NDArray[...] or Optional[NDArray[...]], "
                        f"but got {hint}"
                    )

            if name in alias_t:
                if not self._is_float_hint(hint):
                    raise TypeError(
                        f"In `{func.__name__}`, parameter `{name}` must be float "
                        f"but got {hint}"
                    )

        if "return" not in hints:
            raise TypeError(
                f"In `{func.__name__}`, return type annotation is missing. Must be NDArray[...] or a tuple of NDArrays ."
            )

        if not (
            self._is_ndarray_hint(hints["return"])
            or self._is_tuple_of_ndarrays_hint(hints["return"])
        ):
            raise TypeError(
                f"In `{func.__name__}`, return type must be NDArray[...] or a tuple of NDArrays, "
                f"but got {hints['return']}"
            )

        return func

    def _validate_system_type(self, system_type: str) -> str:
        if not isinstance(system_type, str):
            raise TypeError(
                f"System type must be a string, got {type(system_type).__name__}"
            )
        system_type = system_type.lower()
        if system_type not in self.parent.system_types:
            raise ValueError(
                f"Unrecognized system type '{system_type}'. "
                f"Expected one of: {sorted(self.parent.system_types)}"
            )
        return system_type

    def _resolve_override_func(
        self,
        override_value: Callable | bool | None,
        default_value: Callable,
        fallback_value: Optional[Callable] = None,
    ) -> Callable:
        """
        Resolve an override function with standard logic.

        - If `override_value` is False, return `default_value`.
        - If `override_value` is None and `fallback_value` is provided, return `fallback_value`.
        - Otherwise, validate and return the override function.
        """
        if override_value is False:
            return default_value
        elif override_value is None:
            if fallback_value is not None:
                return fallback_value
        else:
            return self._validate_func_signature(override_value)

    def _resolve_override_str_list(
        self,
        override_value: Sequence[str] | bool,
        default_value: Sequence[str],
    ) -> Sequence[str]:
        """
        Resolve an override for a list of strings (e.g., state/input/measurement names).

        - If `override_value` is False, return `default_value`.
        - Otherwise, validate and return the override list.
        """
        if override_value is False:
            return default_value
        return self._validate_string_sequence(override_value)

    @staticmethod
    def _standardize_time_input_to_linspace(
        t_vector: Optional[NDArray], t_span: Optional[tuple], dt: Optional[float]
    ) -> NDArray:
        if t_vector is not None:
            T = np.asarray(t_vector)
            if T.ndim == 2 and T.shape[1] == 1:
                T = T.ravel()  # Convert (n, 1) → (n,)
            elif T.ndim != 1:
                raise ValueError(
                    "t_vector must be a 1D array or a 2D column vector of shape (n,) or (n, 1)."
                )

            # Check for uniform spacing
            dt_diffs = np.diff(T)
            if not np.allclose(dt_diffs, dt_diffs[0]):
                raise ValueError("t_vector must have uniformly spaced time points.")

            dt = dt_diffs[0]
            t0, tf = T[0], T[-1]
            interval = tf - t0
            n_steps = interval / dt
            N = int(round(n_steps)) + 1
            t_linspace = np.linspace(t0, tf, N)
        else:
            if t_span is None or dt is None:
                raise ValueError(
                    "Either t_vector or both t_span and dt must be provided."
                )
            t0, tf = t_span
            if tf <= t0:
                raise ValueError(
                    "t_span must be in the form (t0, tf) with tf > t0."
                )

            interval = tf - t0
            n_steps = interval / dt

            if not np.isclose(n_steps, round(n_steps)):
                raise ValueError("dt does not divide the time interval evenly.")

            N = int(round(n_steps)) + 1
            t_linspace = np.linspace(t0, tf, N)

        return t_linspace

    @staticmethod
    def _standardize_make_time_windows(
        t_linspace: NDArray[np.float64],
        *,
        divide_time_into_k_windows: Optional[int] = None,
        window_length_in_points: Optional[int] = None,
        window_length_in_time: Optional[float] = None,
        overlap_points_between_windows: Optional[int] = None,
        overlap_time_between_windows: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> List[NDArray[np.float64]]:
        """
        Generate time windows from a given t_linspace with optional overlap.

        Parameters
        ----------
        t_linspace : NDArray
            Time vector to be partitioned into windows.
        divide_time_into_k_windows : int, optional
            Number of windows to divide into. If None, return [t_linspace] as one window.
        window_length_in_points : int, optional
            Length of each window in time steps.
        window_length_in_time : float, optional
            Length of each window in seconds (requires dt).
        overlap_points_between_windows : int, optional
            Overlap between windows in points.
        overlap_time_between_windows : float, optional
            Overlap between windows in seconds (requires dt).
        dt : float, optional
            Time step between t_linspace points, required for time-based inputs.

        Returns
        -------
        List[NDArray]
            List of t_linspace slices representing each window.

        Raises
        ------
        ValueError
            If input combinations are invalid or incompatible with t_linspace.
        """
        n_points = len(t_linspace)

        if divide_time_into_k_windows is None:
            return [t_linspace]

        if (window_length_in_points is not None) and (
            window_length_in_time is not None
        ):
            raise ValueError(
                "Specify only one of `window_length_in_points` or `window_length_in_time`."
            )

        if (overlap_points_between_windows is not None) and (
            overlap_time_between_windows is not None
        ):
            raise ValueError(
                "Specify only one of `overlap_points_between_windows` or `overlap_time_between_windows`."
            )

        # Convert time-based length to point-based
        if window_length_in_time is not None:
            if dt is None:
                raise ValueError(
                    "`dt` must be provided when using `window_length_in_time`."
                )
            p = int(window_length_in_time / dt)
        elif window_length_in_points is not None:
            p = window_length_in_points
        else:
            # Infer p from k
            p = n_points // divide_time_into_k_windows
            if p * divide_time_into_k_windows != n_points:
                valid_ks = [d for d in range(1, n_points + 1) if n_points % d == 0]
                suggestion = ", ".join(str(d) for d in valid_ks)
                raise ValueError(
                    f"Cannot divide time vector of length {n_points} into {divide_time_into_k_windows} equal windows.\n"
                    f"Try one of the following valid values for `divide_time_into_k_windows`: {suggestion}"
                )

        # Convert time-based overlap to point-based
        if overlap_time_between_windows is not None:
            if dt is None:
                raise ValueError(
                    "`dt` must be provided when using `overlap_time_between_windows`."
                )
            m = int(overlap_time_between_windows / dt)
        else:
            m = overlap_points_between_windows or 0

        if m >= p:
            raise ValueError(
                f"Overlap (m={m}) must be smaller than window size (p={p})."
            )

        stride = p - m
        k = divide_time_into_k_windows
        total_required = stride * (k - 1) + p

        if total_required > n_points:
            raise ValueError(
                f"Cannot divide {n_points} points into {k} overlapping windows of length {p} "
                f"with {m} overlap. Required points: {total_required}."
            )

        windows = []
        for i in range(k):
            start = i * stride
            end = start + p
            windows.append(t_linspace[start:end])

        return windows

    def smart_call(
        self,
        func: Callable,
        x: Optional[NDArray] = None,
        u: Optional[NDArray] = None,
        t: Optional[float] = None,
        expected_shape: Optional[tuple[int, int]] = None,
        **extra_kwargs: Any,
    ) -> NDArray:
        """
        Call a function using only the arguments it declares among (x, u, t) plus any
        additional keyword arguments if they match the function signature.

        Examples
        --------
        >>> from pykal_core.utils.safeio import SafeIO
        >>> import numpy as np
        >>> from numpy.typing import NDArray

        >>> safeio = SafeIO()

        """
        sig = inspect.signature(self._validate_func_signature(func))
        call_kwargs = {}

        for name in sig.parameters:
            if name in self._aliases_for_x and x is not None:
                call_kwargs[name] = x
            elif name in self._aliases_for_u and u is not None:
                call_kwargs[name] = u
            elif name in self._aliases_for_t and t is not None:
                call_kwargs[name] = t
            elif name in extra_kwargs:
                call_kwargs[name] = extra_kwargs[name]

        result = func(**call_kwargs)

        if expected_shape is not None:
            if not (
                isinstance(expected_shape, tuple)
                and len(expected_shape) == 2
                and all(isinstance(i, int) for i in expected_shape)
            ):
                raise TypeError("expected_shape must be a tuple of two integers")
            n, m = expected_shape
            if result.shape != (n, m):
                raise ValueError(
                    f"Output shape mismatch. Expected {(n, m)}, got {result.shape}"
                )

        return result
