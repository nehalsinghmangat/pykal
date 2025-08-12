from pykal_core.control_system import Observer
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple


class KF:
    @staticmethod
    def make_L(
        obs: Observer,
        predict: Callable[..., Tuple[NDArray, NDArray]],
        update: Callable[..., Tuple[NDArray, NDArray]],
    ):
        """
        Constructs a loop function L that performs one predict-update step and stores estimates.

        Parameters
        ----------
        obs : Observer
            An observer object with system and estimator.
        predict : Callable
            Prediction function (e.g., predict_EKF).
        update : Callable
            Update function (e.g., update_EKF).
        """

        obs.P_est_hist = []
        obs.x_est_hist = (
            []
        )  ## ahhhhh and then predict_beta can have obs.beta in it for specific saving! fuck yeah, that way shallots wont overwhelm the chickens natural falovars (with its own erase history option for beta!)

        def clear_P_est_and_x_est_history() -> None:
            obs.P_est_hist.clear()
            obs.x_est_hist.clear()

        obs.clear_P_est_and_x_est_history = clear_P_est_and_x_est_history

        def kf(
            yk: NDArray,
            uk: NDArray,
            *,
            f: Callable,
            F: Callable,
            Q: Callable,
            h: Callable,
            H: Callable,
            R: Callable,
            xk: NDArray,
            tk: float,
            Pk: NDArray,
            dt: float,
            **extra_kwargs,
        ) -> NDArray:
            """
            Perform one Kalman filter iteration (predict and update).

            Returns
            -------
            x_upd : NDArray
                Updated state estimate.
            P_upd : NDArray
                Updated covariance estimate.
            """
            x_pred, P_pred = predict(
                obs,
                f=f,
                F=F,
                Q=Q,
                uk=uk,
                xk=xk,
                tk=tk,
                Pk=Pk,
                dt=dt,
                **extra_kwargs,
            )

            x_upd, P_upd = update(
                obs,
                h=h,
                H=H,
                R=R,
                uk=uk,
                yk=yk,
                tk=tk,
                x_pred=x_pred,
                P_pred=P_pred,
            )

            obs.x_est_hist.append(x_upd)
            obs.P_est_hist.append(P_upd)

            return x_upd, P_upd

        return kf

    @staticmethod
    def predict_EKF(
        obs: Observer,
        *,
        f: Callable,
        F: Callable,
        Q: Callable,
        uk: NDArray,
        xk: NDArray,
        tk: float,
        Pk: NDArray,
        dt: float,
        **extra_kwargs,
    ):

        state_names = obs.sys.state_names
        if xk.shape != (len(state_names), 1):
            raise ValueError(
                f"For states {state_names} x0 must have shape ({len(state_names)}, 1), got {xk.shape}"
            )

        n_states = len(state_names)

        Fk = obs.sys.safeio.smart_call(
            F, x=xk, u=uk, t=tk, expected_shape=(n_states, n_states), **extra_kwargs
        )

        Qk = obs.sys.safeio.smart_call(
            Q, x=xk, u=uk, t=tk, expected_shape=(n_states, n_states), **extra_kwargs
        )

        if obs.sys.system_type in ("cti", "ctv"):
            xdot = obs.sys.safeio.smart_call(
                f, x=xk, u=uk, t=tk, expected_shape=(n_states, 1), **extra_kwargs
            )
            x_pred = xk + xdot * dt
            P_pred = Pk + (Fk @ Pk + Pk @ Fk.T + Qk) * dt
        elif obs.sys.system_type in ("dti", "dtv"):
            x_pred = obs.sys.safeio.smart_call(
                f, x=xk, u=uk, t=tk, expected_shape=(n_states, 1), **extra_kwargs
            )
            P_pred = Fk @ Pk @ Fk.T + Qk
        else:
            raise ValueError(
                f"{obs.sys.system_type} is not a valid system_type; must be one of {obs.sys.system_types} "
            )
        return x_pred, P_pred

    @staticmethod
    def update_EKF(
        obs: Observer,
        *,
        h: Callable,
        H: Callable,
        R: Callable,
        uk: NDArray,
        yk: NDArray,
        tk: float,
        x_pred: NDArray,
        P_pred: NDArray,
        **extra_kwargs,
    ):

        state_names = obs.sys.state_names
        measurement_names = obs.sys.measurement_names

        if yk.shape != (len(measurement_names), 1):
            raise ValueError(
                f"For measurements {measurement_names} yk must have shape ({len(measurement_names)}, 1), got {yk.shape}"
            )

        n_states = len(state_names)
        m_meas = len(measurement_names)

        Hk = obs.sys.safeio.smart_call(
            H, x=x_pred, u=uk, t=tk, expected_shape=(m_meas, n_states), **extra_kwargs
        )
        Rk = obs.sys.safeio.smart_call(
            R, x=x_pred, u=uk, t=tk, expected_shape=(m_meas, m_meas), **extra_kwargs
        )
        y_pred = obs.sys.safeio.smart_call(h, x=x_pred, u=uk, t=tk, **extra_kwargs)

        Sk = Hk @ P_pred @ Hk.T + Rk
        ridge = 1e-9 * np.eye(Sk.shape[0])

        try:
            Sk_inv = np.linalg.inv(Sk + ridge)
        except np.linalg.LinAlgError:
            Sk_inv = np.linalg.pinv(Sk + ridge)

        Kk = P_pred @ Hk.T @ Sk_inv
        innovation = yk - y_pred

        I = np.eye(n_states)

        x_upd = x_pred + Kk @ innovation
        P_upd = (I - Kk @ Hk) @ P_pred

        return x_upd, P_upd
