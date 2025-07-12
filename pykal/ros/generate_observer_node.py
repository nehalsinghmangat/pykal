import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
from pykal.observer import Observer
from pykal.system import System
from pykal.ros.system_robots import available_systems
import numpy as np
from numpy.typing import NDArray
from geometry_msgs.msg import Twist
from pykal.utils.extend import (
    pass_partial_update_beta_into_partial_update_override_function,
)
import pandas as pd


class ObserverNode(Node):
    def __init__(self, system: System, dt: float = 0.1):
        super().__init__("observer_node")

        self.sys = system
        self.observer = Observer(system)
        self.dt = dt

        self.x_est = np.zeros((len(system.state_names), 1))
        self.P_est = 0.1 * np.eye(len(system.state_names))
        self.t = 0.0
        self.n = self.x_est.shape[0]

        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )
        self.latest_u = np.zeros((2, 1), dtype=np.float64)

        def cmd_vel_as_u() -> NDArray:
            return self.latest_u

        self.sys.u = cmd_vel_as_u

        self.meas_sub = self.create_subscription(
            Float64MultiArray, "/sys/meas", self.meas_callback, 10
        )
        self.state_pub = self.create_publisher(
            Float64MultiArray, "/observer/state_est", 10
        )

        # Publishers for each beta_i
        #        self.beta_channels = [
        #           self.create_publisher(Float64, f"/observer/beta_{i}", 10)
        #            for i in range(self.n)
        #        ]

        self.ellipse_pubs = [[None for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                topic = f"/observer/ellipse_{i}_{j}"
                self.ellipse_pubs[i][j] = self.create_publisher(
                    Float64MultiArray, topic, 10
                )

        # Live plotting publishers
        self.state_channels = [
            self.create_publisher(Float64, f"/observer/x{i}", 10) for i in range(self.n)
        ]
        self.var_channels = [
            self.create_publisher(Float64, f"/observer/P{i}", 10) for i in range(self.n)
        ]

        # Logging
        self.log = {
            "t": [],
            "x_est": [],
            "P_diag": [],
        }

    def cmd_vel_callback(self, msg: Twist):
        self.latest_u = np.array([[msg.linear.x], [msg.angular.z]], dtype=np.float64)

    def meas_callback(self, msg: Float64MultiArray):
        yk = np.array(msg.data, dtype=np.float64).reshape(-1, 1)

        # EKF predict and update
        self.x_est, self.P_est = self.observer.ekf.predict(
            xk=self.x_est, Pk=self.P_est, dt=self.dt, tk=self.t
        )

        def wrap_beta(Pk: NDArray):
            def beta_ramos_stochastic(*, Pk: NDArray, **kwargs) -> NDArray:
                """
                Compute a clipped and normalized stochastic observability-aware β vector.

                Parameters
                ----------
                P : NDArray
                    Posterior covariance matrix (n x n).

                Returns
                -------
                beta : NDArray
                    Column vector of shape (n, 1) with β_i ∈ [0, 1] and sum β_i = 1.
                """
                try:
                    I = np.linalg.inv(Pk)
                except np.linalg.LinAlgError:
                    I = np.linalg.pinv(Pk)

                I_diag = np.clip(np.diag(I), 0.0, 1.0)
                if np.sum(I_diag) == 0:
                    beta = np.ones_like(I_diag) / len(
                        I_diag
                    )  # fallback: uniform weights
                else:
                    beta = I_diag / np.sum(I_diag)

                return beta.reshape(-1, 1)

            return beta_ramos_stochastic

        self.x_est, self.P_est = self.observer.ekf.update(
            xk=self.x_est,
            Pk=self.P_est,
            yk=yk,
            tk=self.t,
        )

        # Compute and publish beta (Ramos-style)
        #        beta_vec = beta_ramos_stochastic(self.P_est)

        # Publish full state vector
        msg_out = Float64MultiArray(data=self.x_est.flatten().tolist())
        self.state_pub.publish(msg_out)

        # Publish individual state values and variances
        for i in range(self.n):
            x_val = Float64()
            x_val.data = self.x_est[i, 0]
            self.state_channels[i].publish(x_val)

            p_val = Float64()
            p_val.data = self.P_est[i, i]
            self.var_channels[i].publish(p_val)

            #           beta_msg = Float64()
            #            beta_msg.data = float(beta_vec[i, 0])
            #            self.beta_channels[i].publish(beta_msg)

            for j in range(i + 1, self.n):
                xi = self.x_est[i, 0]
                xj = self.x_est[j, 0]
                P_sub = self.P_est[[i, j]][:, [i, j]]  # 2×2 covariance

                ellipse_msg = Float64MultiArray(
                    data=[xi, xj, P_sub[0, 0], P_sub[0, 1], P_sub[1, 1]]
                )
                self.ellipse_pubs[i][j].publish(ellipse_msg)

        # Log
        self.log["t"].append(self.t)
        self.log["x_est"].append(self.x_est.flatten())
        self.log["P_diag"].append(np.diag(self.P_est))
        self.t += self.dt


def main():
    rclpy.init()

    system = available_systems["turtlebot"]
    node = ObserverNode(system=system, dt=0.1)

    try:
        rclpy.spin(node)
    finally:
        # Save logs to CSV
        print("Saving observer log to CSV...")
        df = pd.DataFrame(
            {
                "t": node.log["t"],
                **{f"x{i}": [x[i] for x in node.log["x_est"]] for i in range(node.n)},
                **{f"P{i}": [p[i] for p in node.log["P_diag"]] for i in range(node.n)},
            }
        )
        df.to_csv("observer_log.csv", index=False)

        node.destroy_node()
        rclpy.shutdown()
