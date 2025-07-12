# systems_robots.py
from pykal.system import System
import numpy as np
from numpy.typing import NDArray


def turtle_dynamics(x: NDArray, u: NDArray, t: float) -> NDArray:
    x_pos, y_pos, theta, v, omega = x.flatten()
    v_cmd, omega_cmd = u.flatten()

    dx = np.array(
        [
            [v * np.cos(theta)],
            [v * np.sin(theta)],
            [omega],
            [v_cmd - v],
            [omega_cmd - omega],
        ],
        dtype=np.float64,
    )
    return dx


def full_state_measurement(x: NDArray) -> NDArray:

    return x.reshape(-1, 1)  # full-state observation from /odom


state_names_turtlebot = ["x", "y", "theta", "v", "omega"]
input_names_turtlebot = ["v_cmd", "omega_cmd"]

turtlebot_system = System(
    f=turtle_dynamics,
    h=full_state_measurement,
    state_names=state_names_turtlebot,
    input_names=input_names_turtlebot,
    system_type="cti",
)


# Optional: dictionary for lookup
available_systems = {
    "turtlebot": turtlebot_system,
    # Add others here later
}
