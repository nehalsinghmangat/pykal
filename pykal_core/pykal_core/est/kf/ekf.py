from pykal_core.control_system.system import System
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Callable, Optional, Union, List
from pykal_core.utils import compute


class EKF:
    def __init__(
        self,
        sys: System,
        F: Optional[Callable] = None,
        H: Optional[Callable] = None,
    ):
        self.sys = sys

        self._F = (
            self.sys.safeio._validate_func_signature(F)
            if F is not None
            else compute.Jacobian.wrt_x(self.sys, self.sys.f)
        )
        self._H = (
            self.sys.safeio._validate_func_signature(H)
            if H is not None
            else compute.Jacobian.wrt_x(self.sys, self.sys.h)
        )

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, F):
        self._F = self.sys.safeio._validate_func_signature(F) if F is not None else F

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, H):
        self._H = self.sys.safeio._validate_func_signature(H) if H is not None else H

    def predict(
        self,
        *,
        xk: NDArray,
        Pk: NDArray,
        dt: float,
        tk: float,
        override_system_f: Union[Callable, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_state_names: Union[List[str], bool] = False,
        override_system_input_names: Union[List[str], bool] = False,
        override_system_F: Union[Callable, bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
        override_predict: Union[Callable, bool] = False,
        output_sr: bool = False,
    ) -> Union[pd.Series, tuple[NDArray, NDArray]]:

        state_names = self.sys.safeio._resolve_override_str_list(
            override_system_state_names, self.sys.state_names
        )

        input_names = self.sys.safeio._resolve_override_str_list(
            override_system_input_names, self.sys.input_names
        )

        if xk.shape != (len(state_names), 1):
            raise ValueError(
                f"For states {state_names} x0 must have shape ({len(state_names)}, 1), got {xk.shape}"
            )

        n_states = len(state_names)
        p_inputs = len(input_names)

        f = self.sys.safeio._resolve_override_func(override_system_f, self.sys.f)
        u = self.sys.safeio._resolve_override_func(
            override_system_u, self.sys.u, System.default_u(input_names)
        )
        F = self.sys.safeio._resolve_override_func(override_system_F, self._F)
        Q = self.sys.safeio._resolve_override_func(
            override_system_Q, self.sys.Q, System.zero_Q(state_names)
        )

        uk = self.sys.safeio.smart_call(u, t=tk, expected_shape=(p_inputs, 1))
        Fk = self.sys.safeio.smart_call(
            F, x=xk, u=uk, t=tk, expected_shape=(n_states, n_states)
        )
        Qk = self.sys.safeio.smart_call(
            Q, x=xk, u=uk, t=tk, expected_shape=(n_states, n_states)
        )

        if self.sys.system_type in ("cti", "ctv"):
            xdot = self.sys.safeio.smart_call(f, x=xk, u=uk, t=tk)
            x_pred = xk + xdot * dt

            if override_predict is False:
                P_pred = Pk + (Fk @ Pk + Pk @ Fk.T + Qk) * dt
            else:
                P_pred = self.sys.safeio.smart_call(
                    override_predict,
                    xk=xk,
                    Pk=Pk,
                    dt=dt,
                    tk=tk,
                    Fk=Fk,
                    Qk=Qk,
                    state_names=state_names,
                    input_names=input_names,
                )

        elif self.sys.system_type in ("dti", "dtv"):
            x_pred = self.sys.safeio.smart_call(f, x=xk, u=uk, t=tk)
            P_pred = Fk @ Pk @ Fk.T + Qk

        if output_sr:
            x_df = pd.DataFrame(
                x_pred.T, columns=[f"pred_{name}" for name in state_names], index=[tk]
            )  # shape: (1, n)
            return pd.Series({"x_pred": x_df, "P_pred": P_pred}, name=tk)
        else:
            return x_pred, P_pred

    def update(
        self,
        *,
        xk: NDArray,
        Pk: NDArray,
        yk: NDArray,
        tk: float,
        P0: Optional[NDArray] = None,
        override_system_h: Union[Callable, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_state_names: Union[List[str], bool] = False,
        override_system_measurement_names: Union[List[str], bool] = False,
        override_system_input_names: Union[List[str], bool] = False,
        override_system_H: Union[Callable, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
        override_beta: Union[Callable, bool] = False,
        override_update: Union[Callable, bool] = False,
        output_sr: bool = False,
    ) -> Union[pd.Series, tuple[NDArray, NDArray]]:

        state_names = self.sys.safeio._resolve_override_str_list(
            override_system_state_names, self.sys.state_names
        )
        measurement_names = self.sys.safeio._resolve_override_str_list(
            override_system_measurement_names, self.sys.measurement_names
        )
        input_names = self.sys.safeio._resolve_override_str_list(
            override_system_input_names, self.sys.input_names
        )

        if yk.shape != (len(measurement_names), 1):
            raise ValueError(
                f"For measurements {measurement_names} yk must have shape ({len(measurement_names)}, 1), got {yk.shape}"
            )

        n_states = len(state_names)
        p_inputs = len(input_names)
        m_meas = len(measurement_names)

        h = self.sys.safeio._resolve_override_func(override_system_h, self.sys.h)
        u = self.sys.safeio._resolve_override_func(
            override_system_u, self.sys.u, System.default_u(input_names)
        )
        R = self.sys.safeio._resolve_override_func(
            override_system_R, self.sys.R, System.zero_R(measurement_names)
        )
        H = self.sys.safeio._resolve_override_func(override_system_H, self._H)

        uk = self.sys.safeio.smart_call(u, t=tk, expected_shape=(p_inputs, 1))
        Hk = self.sys.safeio.smart_call(
            H, x=xk, u=uk, t=tk, expected_shape=(m_meas, n_states)
        )
        Rk = self.sys.safeio.smart_call(
            R, x=xk, u=uk, t=tk, expected_shape=(m_meas, m_meas)
        )
        y_pred = self.sys.safeio.smart_call(h, x=xk, u=uk, t=tk)

        Sk = Hk @ Pk @ Hk.T + Rk
        ridge = 1e-9 * np.eye(Sk.shape[0])

        try:
            Sk_inv = np.linalg.inv(Sk + ridge)
        except np.linalg.LinAlgError:
            Sk_inv = np.linalg.pinv(Sk + ridge)

        Kk = Pk @ Hk.T @ Sk_inv
        innovation = yk - y_pred

        if override_update is False:
            x_upd = xk + Kk @ innovation
            I = np.eye(Pk.shape[0])
            P_upd = (I - Kk @ Hk) @ Pk

        else:
            x_upd, P_upd = self.sys.safeio.smart_call(
                override_update,
                xk=xk,
                uk=uk,
                tk=tk,
                Pk=Pk,
                P0=P0,
                Hk=Hk,
                Kk=Kk,
                Rk=Rk,
                state_names=state_names,
                measurement_names=measurement_names,
                input_names=input_names,
                innovation=innovation,
            )

        if output_sr:
            x_df = pd.DataFrame(
                x_upd.T, columns=[f"est_{name}" for name in state_names], index=[tk]
            )  # shape: (1, n)
            return pd.Series({"x_est": x_df, "P_upd": P_upd}, name=tk)
        else:
            return x_upd, P_upd

    def run(
        self,
        *,
        x0: NDArray,
        P0: NDArray,
        Y: Optional[NDArray] = None,
        Y_df: Optional[pd.DataFrame] = None,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        t_vector: Optional[NDArray] = None,
        input_df: Optional[bool] = False,
        output_df: Optional[bool] = False,
        override_system_f: Union[Callable, bool] = False,
        override_system_u: Union[Callable, None, bool] = False,
        override_system_h: Union[Callable, bool] = False,
        override_system_state_names: Union[List[str], bool] = False,
        override_system_input_names: Union[List[str], bool] = False,
        override_system_measurement_names: Union[List[str], bool] = False,
        override_system_F: Union[Callable, bool] = False,
        override_system_H: Union[Callable, bool] = False,
        override_system_R: Union[Callable, None, bool] = False,
        override_system_Q: Union[Callable, None, bool] = False,
        override_predict: Union[Callable, bool] = False,
        override_update: Union[Callable, bool] = False,
    ) -> Union[tuple[pd.DataFrame, pd.Series], tuple[NDArray, NDArray, NDArray]]:
        """
        Run the full EKF estimation over a sequence of measurements.

        Parameters
        ----------
        x0 : NDArray
            Initial state estimate, shape (n, 1).
        P0 : NDArray
            Initial covariance estimate, shape (n, n).
        Y : NDArray
            Measurement history, shape (m, T).
        t_span : tuple[float, float], optional
            Time interval corresponding to measurements.
        dt : float, optional
            Time step.
        t_vector : NDArray, optional
            Optional explicit time vector.
        ...

        Returns
        -------
        x_hist : NDArray
            Filtered state estimates, shape (n, T+1).
        P_hist : NDArray
            Covariance estimates, shape (n, n, T+1).
        t_hist : NDArray
            Time vector used, shape (T+1,).
        """

        # Convert from DataFrame if needed
        if input_df:
            if Y_df is None:
                raise ValueError("Y_df must be provided when input_df=True")
            T = Y_df.index.to_numpy()
            Y = Y_df.to_numpy().T  # shape (n_states, n_steps)
        else:
            if Y is None:
                raise ValueError("Y must be provided when input_df=False")
            if not isinstance(Y, np.ndarray):
                raise TypeError("Y must be a NumPy array")

            T = self.sys.safeio._standardize_time_input_to_linspace(
                t_vector, t_span, dt
            )
            if Y.shape[1] != len(T):
                raise ValueError(
                    f"Y has {Y.shape[1]} steps, but time vector has {len(T)} steps"
                )
        n_states = len(x0)
        num_steps = len(T)
        dtk = T[1] - T[0]

        t_linspace = np.append(T, T[-1] + dtk)
        x_hist_arr = np.empty((n_states, num_steps + 1))
        P_hist_arr = np.empty((n_states, n_states, num_steps + 1))

        xk = x0
        Pk = P0
        x_hist_arr[:, 0] = xk.flatten()
        P_hist_arr[:, :, 0] = Pk

        for k in range(num_steps):
            tk = t_linspace[k]
            yk = Y[:, k].reshape(-1, 1)
            xk_pred, Pk_pred = self.predict(
                xk=xk,
                Pk=Pk,
                dt=dtk,
                tk=tk,
                override_system_f=override_system_f,
                override_system_u=override_system_u,
                override_system_state_names=override_system_state_names,
                override_system_input_names=override_system_input_names,
                override_system_F=override_system_F,
                override_system_Q=override_system_Q,
                override_predict=override_predict,
            )

            xk_upd, Pk_upd = self.update(
                xk=xk_pred,
                Pk=Pk_pred,
                P0=P0,
                yk=yk,
                tk=tk,
                override_system_h=override_system_h,
                override_system_u=override_system_u,
                override_system_measurement_names=override_system_measurement_names,
                override_system_input_names=override_system_input_names,
                override_system_H=override_system_H,
                override_system_R=override_system_R,
                override_update=override_update,
            )

            x_hist_arr[:, k + 1] = xk_upd.flatten()
            P_hist_arr[:, :, k + 1] = Pk_upd

        if output_df:
            state_names = self.sys.safeio._resolve_override_str_list(
                override_system_state_names, self.sys.state_names
            )
            X_est_df = pd.DataFrame(
                x_hist_arr.T,
                index=t_linspace,
                columns=[f"est_{name}" for name in state_names],
            )

            X_est_df.index.name = "time"

            P_hist_sr = pd.Series(
                {t: P_hist_arr[:, :, i] for i, t in enumerate(t_linspace)}
            )
            return X_est_df, P_hist_sr

        return x_hist_arr, P_hist_arr, t_linspace

    def override_update_partial(self, beta: Callable, beta_log: Optional[pd.DataFrame]):

        def partial_update_dynamic_beta(
            *,
            xk: NDArray,
            uk: NDArray,
            tk: float,
            Pk: NDArray,
            P0: NDArray,
            Hk: NDArray,
            Kk: NDArray,
            Rk: NDArray,
            state_names: List[str],
            measurement_names: List[str],
            input_names: List[str],
            innovation: NDArray,
        ) -> tuple[NDArray, NDArray]:

            beta_vector = self.sys.safeio.smart_call(
                beta,
                xk=xk,
                uk=uk,
                tk=tk,
                Pk=Pk,
                P0=P0,
                Hk=Hk,
                Kk=Kk,
                Rk=Rk,
                state_names=state_names,
                measurement_names=measurement_names,
                input_names=input_names,
                innovation=innovation,
            )

            if beta_log is not None:
                beta_vector_sr = pd.Series(beta_vector.flatten(), index=state_names)
                beta_log.loc[tk] = beta_vector_sr
            beta_matrix = np.diagflat(beta_vector)
            I = np.eye(Pk.shape[0])
            x_upd = xk + beta_matrix @ Kk @ innovation
            P_upd = (I - beta_matrix @ Kk @ Hk) @ Pk @ (
                I - beta_matrix @ Kk @ Hk
            ).T + beta_matrix @ Kk @ Rk @ Kk.T @ beta_matrix.T
            return x_upd, P_upd

        return partial_update_dynamic_beta
