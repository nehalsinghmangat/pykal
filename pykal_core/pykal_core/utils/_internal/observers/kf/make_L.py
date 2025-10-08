from pykal_core.control_system import ObserverBlock
from pykal_core.utils import compute
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, Optional


class KF:
    @staticmethod
    def make_L(
        obs_block: ObserverBlock,
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
        obs_block.P_pred_hist = []
        obs_block.P_upd_hist = []
        obs_block.x_pred_hist = []
        obs_block.x_upd_hist = []
        obs_block.dt = 0.1

        def kf(
            yk: NDArray,
            uk: NDArray,
            tk: float,
            *,
            f: Callable,
            h: Callable,
            Q: Callable,
            R: Callable,
            F: Optional[Callable] = None,
            H: Optional[Callable] = None,
            **extra_kwargs,
        ) -> NDArray:
            """
            Perform one Kalman filter iteration (predict and update).

            Returns
            -------
            x_upd : NDArray
                Updated state estimate.
            """
            x_pred, P_pred = predict(
                obs_block,
                tk,
                uk,
                f,
                Q,
                F=F,
                **extra_kwargs,
            )

            x_upd, P_upd = update(
                obs_block,
                h=h,
                H=H,
                R=R,
                uk=uk,
                yk=yk,
                tk=tk,
                x_pred=x_pred,
                P_pred=P_pred,
            )

            obs_block.x_pred_hist.append(x_pred)
            obs_block.P_pred_hist.append(P_pred)
            obs_block.x_upd_hist.append(x_upd)
            obs_block.P_upd_hist.append(P_upd)

            return x_upd

        return kf
