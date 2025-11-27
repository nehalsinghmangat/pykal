import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpy.random import Generator
from typing import Any,Callable, Dict, Optional, List, Literal, Tuple, Mapping
from pykal import safeio
from scipy.integrate import solve_ivp
from pykal.safeio import SafeIO
from pykal.block_dynamical_system import BlockDynamicalSystem as BlockDS


class Simulation:
    @staticmethod
    def of_dynamical_system_block(
        block: BlockDS,
        *,
        f: Optional[Callable] = None,
        h: Optional[Callable] = None,
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
        x0: Optional[ArrayLike] = None,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        tk_in_sim: Optional[float] = None,
        func_params: Optional[Mapping[str, Dict[str, Any]]] = None,
        param_params: Optional[Mapping[str, Dict[str, Any]]] = None,            
        rng: Optional[Generator] = None,
    ):
        # ---- Standardize Inputs --------------------------------------------------------
        t_linspace = SafeIO._standardize_time_input_to_linspace(
            t_vector, t_span, tk_in_sim, dt
        )

        def standardize_input_dictionary(input_dict: Dict, t_linspace: NDArray):
            """
            Broadcast and standardize time-dependent inputs.
            """
            output_dict = {}

            for k, v in input_dict.items():
                output_dict[k] = [v] * len(t_linspace)

            return output_dict
        
        def key_values_at_tk(input_dict: Dict, tk: float, t_linspace: NDArray) -> Dict[str, object]:
            def index_of_t(t: float, t_linspace: NDArray) -> int:
                i = int(np.searchsorted(t_linspace, t, side="right") - 1)
                return 0 if i < 0 else (len(t_linspace) - 1 if i >= len(t_linspace) else i)

            idx = index_of_t(tk, t_linspace)
            out = {}
            for k, seq in input_dict.items():
                val = seq[idx]
                # unwrap 0-D object arrays (e.g., array(<function ...>, dtype=object))
                if isinstance(val, np.ndarray) and val.dtype == object and val.ndim == 0:
                    val = val.item()
                out[k] = val
            return out

        merged_params, _ = SafeIO.merge_named_params(func_params,strict=True)

        merged_params["param_params"] = param_params

        kwarg_dict_over_time = (
            standardize_input_dictionary(merged_params, t_linspace)
            if func_params is not None
            else {}
        )

        if h is None:
            h = block.h

        if f is None:
            f = block.f
            # ---- Simulate Y with no dynamics --------------------------------------------------------
            if f.__name__ == "zero_dynamics":
                if tk_in_sim is not None:  # return single step h value
                    return SafeIO.smart_call(
                        h,
                        t=tk_in_sim,
                        kwargs_dict=key_values_at_tk(
                            kwarg_dict_over_time, tk_in_sim, t_linspace
                        ),
                    )
                else:  # simulate over time
                    Y = []
                    for tk in t_linspace:
                        yk = SafeIO.smart_call(
                            h,
                            t=tk,
                            kwargs_dict=key_values_at_tk(
                                kwarg_dict_over_time, tk, t_linspace
                            ),
                        )
                        yk = np.asarray(yk).reshape(-1)  # ensure (m,)
                        Y.append(yk)
                    return Y

            else:

                # ---- Simulate X --------------------------------------------------------
                if x0 is None:
                    raise ValueError("x0 must be provided if f is not None.")

                if block.sys_type in {"cti", "ctv"}:

                    if not isinstance(x0, np.ndarray):
                        raise ValueError(
                            "For a continuous system, we use solve_ivp to simulate dynamics. x0 must be NDArray."
                        )

                    def wrapped_f(tk: float, xk: NDArray) -> NDArray:
                        xk = np.asarray(xk).reshape(-1)
                        dx = SafeIO.smart_call(
                            f,
                            x=xk,
                            t=float(tk),
                            kwargs_dict=key_values_at_tk(
                                kwarg_dict_over_time, tk, t_linspace
                            ),
                        )
                        return np.asarray(dx).reshape(-1)  # (n,)

                    t0 = float(t_linspace[0])
                    tf = float(t_linspace[-1]) if tk_in_sim is None else tk_in_sim + dt
                    t_eval = t_linspace if tk_in_sim is None else np.asarray([tk_in_sim + dt])

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
                    x_last = x0
                    X = []
                    for tk in t_linspace:
                        xk = x_last  # last row
                        x_next = SafeIO.smart_call(
                            f,
                            x=xk,
                            t=tk,
                            kwargs_dict=key_values_at_tk(
                                kwarg_dict_over_time, tk, t_linspace
                            ),
                        )
                        X.append(x_next)
                        x_last = x_next


                else:
                    raise ValueError(
                        f'{block.sys_type} must be one of "cti", "ctv", "dti", "dtv"'
                    )

            if rng is None:
                 rng = np.random.default_rng(42)
            # ---- Add process noise --------------------------------------------------------

            if Q is None:
                Q = block.Q
            if Q is not None:
                X_noisy = []
                for xk, tk in zip(X, t_linspace):
                    Qk = SafeIO.smart_call(
                        Q,
                        x=xk,
                        t=tk,
                        kwargs_dict=key_values_at_tk(
                            kwarg_dict_over_time, tk, t_linspace
                        ),
                    )
                    xk_noisy = xk + rng.multivariate_normal(
                        mean=np.zeros(xk.shape[0]), cov=Qk
                    )
                    X_noisy.append(xk_noisy)
                X = X_noisy

            # ---- Simulate Y --------------------------------------------------------
            Y = []
            for xk, tk in zip(X, t_linspace):
                yk = SafeIO.smart_call(
                    h,
                    x=xk,
                    t=tk,
                    kwargs_dict=key_values_at_tk(kwarg_dict_over_time, tk, t_linspace),
                )
                Y.append(yk)
            # ---- Add Output Noise --------------------------------------------------------
            if R is None:
                R = block.R
            if R is not None:
                Y_noisy = []
                for yk, tk in zip(Y, t_linspace):
                    Rk = SafeIO.smart_call(
                        R,
                        x=yk,
                        t=tk,
                        kwargs_dict=key_values_at_tk(
                            kwarg_dict_over_time, tk, t_linspace
                        ),
                    )
                    yk_noisy = yk + rng.multivariate_normal(
                        mean=np.zeros(yk.shape[0]), cov=Rk
                    )
                    Y_noisy.append(yk_noisy)
                Y = Y_noisy

            if tk_in_sim is not None:
                return X[0], Y[0]
            else:
                return X,Y
