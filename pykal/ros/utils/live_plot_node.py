import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


class LivePlot(Node):
    def __init__(self, n_states: int = 5):
        super().__init__("live_plot")
        self.n = n_states

        # Each pair (i, j) → (data, fig, ax)
        self.ellipses = {}

        # Create subscriptions and individual figures for all unique state pairs
        for i in range(self.n):
            for j in range(i + 1, self.n):
                topic = f"/observer/ellipse_{i}_{j}"
                self.create_subscription(
                    Float64MultiArray, topic, self.make_callback(i, j), 10
                )

                fig, ax = plt.subplots()
                ax.set_title(f"3σ Ellipse: x{i} vs x{j}")
                ax.set_xlabel(f"x{i}")
                ax.set_ylabel(f"x{j}")
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_aspect("equal")
                ax.grid(True)

                self.ellipses[(i, j)] = {
                    "data": None,
                    "fig": fig,
                    "ax": ax,
                }

        plt.ion()
        plt.show()

    def make_callback(self, i: int, j: int):
        def callback(msg):
            data = np.array(msg.data)
            if data.shape[0] == 5:
                self.ellipses[(i, j)]["data"] = data
                self.update_plot(i, j)

        return callback

    def update_plot(self, i: int, j: int):
        entry = self.ellipses[(i, j)]
        data = entry["data"]
        ax = entry["ax"]

        if data is None:
            return

        xi, xj, Pii, Pij, Pjj = data
        cov = np.array([[Pii, Pij], [Pij, Pjj]])

        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(f"3σ Ellipse: x{i} vs x{j}")
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        ax.grid(True)

        ax.plot(xi, xj, "bx", label="Estimate")
        self.plot_covariance_ellipse(ax, xi, xj, cov, label="3σ Ellipse")

        ax.legend()
        entry["fig"].canvas.draw()
        entry["fig"].canvas.flush_events()

    def plot_covariance_ellipse(self, ax, x, y, cov, label, n_std=3.0):
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)

        ellipse = Ellipse(
            xy=(x, y),
            width=width,
            height=height,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            label=label,
        )
        ax.add_patch(ellipse)


def main():
    rclpy.init()
    node = LivePlot(n_states=5)
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()
