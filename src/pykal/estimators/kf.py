import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple

class KF:
    @staticmethod
    def kf_f(
        xhat_P: Tuple[NDArray, NDArray],  # [x̂, P]
        *,
        yk: NDArray,
        f: Callable,
        f_params: Dict,
        F: Callable,
        F_params: Dict,            
        Q: Callable,
        Q_params: Dict,
        h: Callable,
        h_params: Dict,            
        H: Callable,
        H_params: Dict,                        
        R: Callable,
        R_params: Dict,                                    
    ) -> Tuple[NDArray, NDArray]:
        """
        Extended Kalman Filter step (predict + update) using column-vector convention internally.

        Shapes:
            x̂_k : (n, 1)
            P_k  : (n, n)
            y_k  : (m, 1)
            F,H  : (n,n), (m,n)
            Q,R  : (n,n), (m,m)
        """

        # extract state variables
        _, Pk = xhat_P

        # predict
        x_pred = f(**f_params)        
        Fk = F(**F_params)
        Qk = Q(**Q_params)
        P_pred = Fk @ Pk @ Fk.T + Qk        

        # innovation
        y_pred = h(**h_params)
        innovation = yk - y_pred                       # (m,1)

        # update
        Hk = H(**H_params)
        Rk = R(**R_params)
        Sk = Hk @ P_pred @ Hk.T + Rk                       # (m,m)
        ridge = 1e-9 * np.eye(Sk.shape[0])
        try:
            Sk_inv = np.linalg.inv(Sk + ridge)
        except np.linalg.LinAlgError:
            Sk_inv = np.linalg.pinv(Sk + ridge)

        Kk = P_pred @ Hk.T @ Sk_inv                        # (n,m)        
        x_upd = x_pred + Kk @ innovation                   # (n,1)
        I = np.eye(P_pred.shape[0])
        P_upd = (I - Kk @ Hk) @ P_pred @ (I - Kk @ Hk).T + Kk @ Rk @ Kk.T
        P_upd = 0.5 * (P_upd + P_upd.T)                    # symmetrize

        return (x_upd, P_upd)
    @staticmethod
    def kf_h(xhat_P: Tuple[NDArray, NDArray]) -> NDArray:
        # extracts current state estimate
        return xhat_P[0]

