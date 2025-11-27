import numpy as np
from numpy.typing import NDArray, ArrayLike
import inspect
from collections import ChainMap
from typing import (
    Any,
    Callable,
    Optional,
    Mapping,
    get_type_hints,
    get_origin,
    Sequence,
    get_args,
    Union,
    List,
    Tuple,
    Dict,
)


class SafeIO:

    _ALIASES_FOR_X = {"x", "xk", "state", "x_vec"}
    _ALIASES_FOR_U = {"u", "uk", "input", "control", "u_vec"}
    _ALIASES_FOR_T = {"t", "tk", "time"}

    @staticmethod
    def as_col(v: NDArray) -> NDArray:
        v = np.asarray(v)
        if v.ndim == 1:
            return v.reshape(-1, 1)
        if v.ndim == 2:
            if v.shape[1] == 1:
                return v
            if v.shape[0] == 1:
                return v.T
        raise ValueError(f"Expected vector-like array; got shape {v.shape}")
    
    @staticmethod
    def same_orientation(v_col: NDArray, like: NDArray) -> NDArray:
        like = np.asarray(like)
        if like.ndim == 1:
            return v_col.reshape(-1)
        if like.ndim == 2 and like.shape[0] == 1:
            return v_col.T
        return v_col    

    @classmethod
    def smart_call(
        cls,
        func: Callable[..., Any],
        x: Optional[np.ndarray] = None,
        t: Optional[float] = None,
        kwargs_dict: Optional[Dict] = None,
    ):
        sig = inspect.signature(func)
        params = sig.parameters
        pool: dict[str, Any] = {}
        if kwargs_dict:
            pool.update(kwargs_dict)

        def first_match(aliases: set[str]) -> str | None:
            for name in params:
                if name in aliases:
                    return name
            return None

        # Route x/t to the callee’s actual parameter names if present
        if x is not None:
            name_x = first_match(cls._ALIASES_FOR_X)
            if name_x and name_x not in pool:
                pool[name_x] = x
        if t is not None:
            name_t = first_match(cls._ALIASES_FOR_T)
            if name_t and name_t not in pool:
                pool[name_t] = t

        # Build positional list for any positional-only params (declared with '/')
        pos_args: list[Any] = []
        for name, p in params.items():
            if p.kind is inspect.Parameter.POSITIONAL_ONLY and name in pool:
                pos_args.append(pool.pop(name))

        # Build kwargs for positional-or-keyword and keyword-only params
        call_kwargs: dict[str, Any] = {}
        for name, p in params.items():
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                if name in pool:
                    call_kwargs[name] = pool.pop(name)

        # Pass through unknown keys only if func accepts **kwargs
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            call_kwargs.update(pool)  # remaining names
        # else: drop leftovers silently

        # Validate duplicates/misbindings, then call
        bound = sig.bind_partial(*pos_args, **call_kwargs)
        out = func(*bound.args, **bound.kwargs)

        return out    
    @staticmethod
    def _standardize_time_input_to_linspace(
        t_vector: Optional[ArrayLike],
        t_span: Optional[Tuple[float, float]],
        tk: Optional[float],
        dt: Optional[float],
    ) -> NDArray:
        """
        Standardize user-provided time specifications into a clean, uniformly spaced 1D array.

        This utility enforces that time is always represented as a uniformly spaced
        NumPy array (via ``np.linspace``). It accepts different input formats:

        1. **Explicit vector of times** (`t_vector`):
           - Accepts any array-like input (list, tuple, NumPy array, Pandas Series, etc.).
           - Internally converted to a NumPy array.
           - Shape can be ``(n,)`` or column vector ``(n, 1)``.
           - Must be uniformly spaced.
           - Returns a standardized uniform grid from ``t_vector[0]`` to ``t_vector[-1]``.

        2. **Time span and step** (`t_span`, `dt`):
           - ``t_span = (t0, tf)`` with ``tf > t0``.
           - ``dt`` must evenly divide the interval ``tf - t0``.
           - Returns a uniform grid spanning from ``t0`` to ``tf``.

        3. **Single start time and step** (`tk`, `dt`):
           - Returns a 2-element array: ``[tk, tk + dt]``.

        If none of these conditions are satisfied, a ``ValueError`` is raised.

        Parameters
        ----------
        t_vector : array_like, optional
            Explicit time vector, convertible to a NumPy array.
            Must be 1D ``(n,)`` or 2D column vector ``(n,1)``, and uniformly spaced.
            If provided, takes precedence over other inputs.
        t_span : tuple, optional
            Tuple ``(t0, tf)`` defining the simulation interval. Requires ``dt``.
        tk : float, optional
            Single starting time. Requires ``dt``. Returns ``[tk, tk + dt]``.
        dt : float, optional
            Time step size. Required if using ``t_span`` or ``tk``.

        Returns
        -------
        np.ndarray
            1D NumPy array of uniformly spaced time points.

        Raises
        ------
        ValueError
            If input formats are invalid, inconsistent, or not uniformly spaced.

        Examples
        --------
        >>> import numpy as np
        >>> # Using t_vector
        >>> t_vec = np.array([[0.0],[0.1],[0.2],[0.3]])
        >>> Standardizer._standardize_time_input_to_linspace(t_vec, None, None, None)
        array([0. , 0.1, 0.2, 0.3])

        >>> # Using t_span and dt
        >>> Standardizer._standardize_time_input_to_linspace(None, (0.0, 1.0), None, 0.25)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])

        >>> # Using tk and dt
        >>> Standardizer._standardize_time_input_to_linspace(None, None, 0.5, 0.1)
        array([0.5, 0.6])
        """
        if t_vector is not None:
            T = np.asarray(t_vector)
            if T.ndim == 2 and T.shape[1] == 1:
                T = T.ravel()  # Convert (n, 1) → (n,)
            elif T.ndim != 1:
                raise ValueError(
                    "t_vector must be ArrayLike and coercible into a np.ndarray of shape (n,) or (n,1)."
                )

            # Check for uniform spacing
            dt_diffs = np.diff(T)
            if not np.allclose(dt_diffs, dt_diffs[0]):
                raise ValueError("t_vector must have uniformly spaced time points.")

            # turn T into np.linspace
            dt = dt_diffs[0]
            t0, tf = T[0], T[-1]
            interval = tf - t0
            n_steps = interval / dt
            N = int(round(n_steps)) + 1
            t_linspace = np.linspace(t0, tf, N)
            return t_linspace

        elif t_span is not None:
            if dt is None:
                raise ValueError("If t_span is provided, dt must be provided.")
            t0, tf = t_span
            if tf <= t0:
                raise ValueError("t_span must be in the form (t0, tf) with tf > t0.")

            interval = tf - t0
            n_steps = interval / dt

            if not np.isclose(n_steps, round(n_steps)):
                raise ValueError("dt does not divide the t_span interval evenly.")

            N = int(round(n_steps)) + 1
            t_linspace = np.linspace(t0, tf, N)
            return t_linspace

        elif tk is not None:
            if dt is None:
                raise ValueError("If tk is provided, dt must be provided.")
            t_linspace = np.linspace(tk, tk + dt, 1)  # array([tk])
            return t_linspace

        else:
            raise ValueError("Provide one of: t_vector; t_span and dt; or tk and dt.")

    @staticmethod
    def merge_named_params(
            func_params: Optional[Mapping[str, Dict[str, Any]]],
            *,
            strict: bool = False
        ) -> tuple[dict[str, Any], dict[str, str]]:
            """
            Merge a dict of dicts preserving insertion order precedence.
            Later (rightmost) names have higher precedence.

            Returns (merged, provenance).
            """
            func_params = func_params or {}
            names = list(func_params.keys())  # insertion order preserved

            # STRICT: complain if the same key appears in multiple sub-dicts
            if strict:
                seen_from: dict[str, str] = {}
                for name in names:
                    for k in func_params[name].keys():
                        if k in seen_from:
                            raise ValueError(f"key '{k}' appears in both '{seen_from[k]}' and '{name}'")
                        seen_from[k] = name

            # Build a view where later names override earlier ones
            view = ChainMap(*[func_params[n] for n in reversed(names)])
            merged: Dict[str, Any] = dict(view)

            # Provenance: who set each winning key
            provenance: Dict[str, str] = {}
            for k in merged:
                for name in reversed(names):  # check from highest precedence
                    if k in func_params[name]:
                        provenance[k] = name
                        break

            return merged, provenance
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
