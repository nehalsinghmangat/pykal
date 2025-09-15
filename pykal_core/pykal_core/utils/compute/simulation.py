import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Callable, Dict, Optional, List, Literal
from scipy.integrate import solve_ivp
from pykal_core.utils.control_system.safeio import SafeIO
from pykal_core.blocks import ControlBlock as CB


class Simulation:
    @staticmethod
    def of_dynamical_system_block(
        block: CB.DSBlock,
        *,
        f: Optional[Callable] = None,
        h: Optional[Callable] = None,
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
        x0: Optional[ArrayLike] = None,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        tk: Optional[float] = None,
        arg_dict: Optional[Dict[str, ArrayLike]] = None,
        kwarg_dict: Optional[Dict[str, ArrayLike]] = None,
        rng_seed: Optional[int] = 42,
    ):

        # ---- Helper Functions --------------------------------------------------------

        def standardize_input_dictionaries(input_dict: Dict):
            output_dict = {}

            for k, v in input_dict.items():
                arr = np.asarray(v)
                if arr.ndim == 0:
                    arr = np.tile(
                        arr, (len(t_linspace), 1)
                    )  # broadcast scalar as (1,) array over time
                if arr.ndim == 1:
                    arr = np.tile(
                        arr, (len(t_linspace), 1)
                    )  # broadcast vector over time
                if arr.shape[0] != len(t_linspace):
                    raise ValueError(
                        f'"{k}" must have first dim T={len(t_linspace)}; got {arr.shape}'
                    )
                output_dict[k] = arr
            return output_dict

        def index_of_t(t: float) -> int:
            i = int(np.searchsorted(t_linspace, t, side="right") - 1)
            return 0 if i < 0 else (len(t_linspace) - 1 if i >= len(t_linspace) else i)

        def key_val_at_t(input_dict: Dict, t: float) -> Dict[str, NDArray]:
            idx = index_of_t(t)
            out = {}
            for k, arr in input_dict.items():
                out[k] = np.asarray(arr[idx])
            return out

        # --- create RNG once ----------------------------------------------------------
        rng = np.random.default_rng(rng_seed)

        def add_noise(
            xk: NDArray,
            arg_dict: Dict,
            kwarg_dict: Dict,
            covariance_matrix_function: Callable,
            tk: float,
        ):
            # Always work with 1-D state/output vectors for noise math
            xk = np.asarray(xk).reshape(-1)
            args = key_val_at_t(arg_dict, tk)
            kwargs = key_val_at_t(kwarg_dict, tk)
            Mk = SafeIO.smart_call(
                covariance_matrix_function,
                x=xk,
                t=float(tk),
                extra_args_dict=args,
                extra_kwargs_dict=kwargs,
            )
            muk = rng.multivariate_normal(mean=np.zeros(xk.shape[0]), cov=Mk)
            return (xk + muk).reshape(-1)  # keep 1-D row

        def call_function_with_args_and_kwargs(
            func: Callable[..., NDArray],
            tk: float,
            arg_dict: Dict[str, NDArray],
            kwarg_dict: Dict[str, NDArray],
            xk: Optional[NDArray] = None,  # <-- fix typing & default
        ) -> NDArray:
            args = key_val_at_t(arg_dict, tk)
            kwargs = key_val_at_t(kwarg_dict, tk)
            out = SafeIO.smart_call(
                func, t=float(tk), x=xk, extra_args_dict=args, extra_kwargs_dict=kwargs
            )
            return np.asarray(out).reshape(-1)  # normalize to 1-D row

        # ---- Standardize Inputs --------------------------------------------------------

        t_linspace = SafeIO._standardize_time_input_to_linspace(
            t_vector, t_span, tk, dt
        )
        arg_dict = (
            standardize_input_dictionaries(arg_dict) if arg_dict is not None else {}
        )
        kwarg_dict = (
            standardize_input_dictionaries(kwarg_dict) if kwarg_dict is not None else {}
        )

        # ---- Shortcuts --------------------------------------------------------

        def identity_map(xk: NDArray) -> NDArray:
            return xk

        if h is None:
            h = block.h

            if h is None:

                h = identity_map

        if f is None:
            f = block.f

            if f is None:
                if tk is not None:
                    return call_function_with_args_and_kwargs(
                        h, tk, arg_dict, kwarg_dict
                    )
                else:
                    Y_list = []
                    for tk in t_linspace:
                        yk = call_function_with_args_and_kwargs(
                            h, tk, arg_dict, kwarg_dict
                        )
                        yk = np.asarray(yk).reshape(-1)  # ensure 1D
                        Y_list.append(yk)
                    Y = np.vstack(Y_list)  # shape (len(t_linspace), m)
                    return Y

            else:

                # ---- Simulate X --------------------------------------------------------
                if x0 is None:
                    raise ValueError("x0 must be provided if f is not None.")
                else:
                    x0 = np.asarray(x0)

                    # If it's 1D, reshape to column
                    if x0.ndim == 1:
                        x0 = x0.reshape(-1, 1)

                    # If it's 2D but not column shaped, also fix
                    elif x0.ndim == 2 and x0.shape[1] != 1:
                        x0 = x0.reshape(-1, 1)

                    # Optional: sanity check
                    if x0.ndim != 2 or x0.shape[1] != 1:
                        raise ValueError(
                            f"x0 must be column vector of shape (n,1), got {x0.shape}"
                        )

                if block.sys_type in {"cti", "ctv"}:

                    def wrapped_f(tk: float, xk: NDArray) -> NDArray:
                        xk = np.asarray(xk).reshape(-1)
                        args = key_val_at_t(arg_dict, tk)
                        kwargs = key_val_at_t(kwarg_dict, tk)
                        dx = SafeIO.smart_call(
                            f,
                            x=xk,
                            t=float(tk),
                            extra_args_dict=args,
                            extra_kwargs_dict=kwargs,
                        )
                        return np.asarray(dx).reshape(-1)  # (n,)

                    t0 = float(t_linspace[0])
                    tf = float(t_linspace[-1]) if tk is None else float(tk + dt)
                    t_eval = t_linspace if tk is None else np.array([tf], dtype=float)

                    sol = solve_ivp(
                        fun=wrapped_f,
                        t_span=(t0, tf),
                        t_eval=t_eval,
                        y0=x0.reshape(-1),
                        vectorized=False,
                    )
                    if not sol.success:
                        raise RuntimeError(f"solve_ivp failed: {sol.message}")

                    X = sol.y.T  # shape (T,n) with rows 1-D

                elif block.sys_type in {"dti", "dtv"}:

                    X_rows: List[NDArray] = []
                    X_rows.append(np.asarray(x0).reshape(-1))
                    for k in range(len(t_linspace) - 1):
                        tk_i = float(t_linspace[k])
                        xk = X_rows[-1]  # last row
                        args = key_val_at_t(arg_dict, tk_i)
                        kwargs = key_val_at_t(kwarg_dict, tk_i)
                        x_next = SafeIO.smart_call(
                            f,
                            x=xk,
                            t=tk_i,
                            extra_args_dict=args,
                            extra_kwargs_dict=kwargs,
                        )
                        X_rows.append(np.asarray(x_next).reshape(-1))

                    X = np.vstack(X_rows)  # (T, n)
                    if tk is not None:
                        X = X[:-1]  # keep up to current step if single-step query

                else:
                    raise ValueError(
                        f'{block.sys_type} must be one of "cti", "ctv", "dti", "dtv"'
                    )

            # ---- Add process noise --------------------------------------------------------

            if Q is None:
                Q = block.Q
            if Q is not None:
                X_noisy = [
                    add_noise(xk, arg_dict, kwarg_dict, Q, float(tk_i))
                    for xk, tk_i in zip(X, t_linspace)
                ]
                X = np.vstack(X_noisy)

            if h is identity_map:
                return X

            # ---- Simulate Y --------------------------------------------------------
            Y = []
            for xk, tk in zip(X, t_linspace):
                yk = call_function_with_args_and_kwargs(
                    h, tk, arg_dict, kwarg_dict, xk=xk
                )
                Y.append(yk)
            Y = np.vstack(Y)

            # ---- Add Output Noise --------------------------------------------------------
            if R is None:
                R = block.R
            if R is not None:
                Y_noisy = [
                    add_noise(yk, arg_dict, kwarg_dict, R, float(tk_i))
                    for yk, tk_i in zip(Y, t_linspace)
                ]
                Y = np.vstack(Y_noisy)

            return X, Y
