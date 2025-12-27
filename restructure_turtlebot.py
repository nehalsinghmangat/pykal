#!/usr/bin/env python3
"""Restructure turtlebot notebook to match car_cruise_control pedagogical approach."""

import json
import sys


def restructure_turtlebot_notebook():
    """Restructure turtlebot to match car_cruise_control pattern."""

    nb_path = "docs/source/notebooks/tutorial/theory_to_python/turtlebot_state_estimation.ipynb"

    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # New cell structure following car_cruise_control pattern
    new_cells = []

    # Cell 0: Parent link (keep existing)
    new_cells.append({
        "cell_type": "markdown",
        "id": "8b486ccf",
        "metadata": {},
        "source": [
            "[← Control Systems as Dynamical Systems](../../../getting_started/theory_to_python/control_systems_as_dynamical_systems.rst)"
        ]
    })

    # Cell 1: Title
    new_cells.append({
        "cell_type": "markdown",
        "id": "24311b13",
        "metadata": {},
        "source": [
            "# Example: TurtleBot with Noisy Odometry"
        ]
    })

    # Cell 2: Intro paragraph + System Overview
    new_cells.append({
        "cell_type": "markdown",
        "id": "102cb17b",
        "metadata": {},
        "source": [
            "Suppose we want to navigate a mobile robot through an environment. We might need it to follow a path, visit waypoints, or track a moving target. To accomplish this, we need accurate knowledge of the robot's position and orientation.\n\n",
            "However, our odometry sensor (wheel encoders) is noisy. Uneven terrain, wheel slip, and measurement quantization introduce errors that accumulate over time. In this notebook, we take a classic mobile robotics problem -- a TurtleBot operating with noisy odometry -- and use pykal to recast it as a composition of interacting dynamical systems.\n\n",
            "## System Overview\n\n",
            "We model the robot dynamics, implement sensor noise, design the observer, and integrate everything into a complete feedback system. This notebook bridges theory and practice for ground robot state estimation."
        ]
    })

    # Cell 3: Block Diagram
    new_cells.append({
        "cell_type": "markdown",
        "id": "e42c0937",
        "metadata": {},
        "source": [
            "## Block Diagram\n\n",
            "We can model the TurtleBot3 feedback system we are interested in as follows:\n\n",
            "![TurtleBot feedback system](../../../_static/tutorial/theory_to_python/turtlebot_feedback_system.svg)\n\n",
            "where **waypoint generator** produces reference trajectories (e.g., move to coordinates), **velocity command** is a high-level controller that outputs linear and angular velocity, **TurtleBot** is our differential-drive robot (our plant), **odometry sensor** is the wheel encoders measuring position and orientation, and **Kalman Filter** is the state observer that fuses control inputs and noisy measurements.\n\n",
            ":::{note}\n",
            "The TurtleBot has an onboard low-level controller that converts velocity commands to wheel speeds. Our \"controller\", then, is simply the high-level velocity command generator.\n",
            ":::"
        ]
    })

    # Cell 4: Discrete-time Dynamical Systems
    new_cells.append({
        "cell_type": "markdown",
        "id": "c48de5e5",
        "metadata": {},
        "source": [
            "### Discrete-time Dynamical Systems\n\n",
            "We cast each component as a discrete-time dynamical system:\n\n",
            "![TurtleBot as composition of dynamical systems](../../../_static/tutorial/theory_to_python/turtlebot_composition_of_dynamical_systems.svg)\n\n",
            "We discuss the derivation of each dynamical system block below.\n\n",
            ":::{note}\n",
            "Unlike the cruise control example, the TurtleBot has **nonlinear** kinematics (due to the orientation-dependent motion). This makes the Kalman filter an Extended Kalman Filter (EKF), linearizing around the current estimate at each step.\n",
            ":::"
        ]
    })

    # Cell 5: Block 1 - Waypoint Generator (markdown)
    new_cells.append({
        "cell_type": "markdown",
        "id": "673b2724",
        "metadata": {},
        "source": [
            "### Block 1: Waypoint Generator\n\n",
            "<div style=\"text-align: center;\">\n",
            " <img src=\"../../../_static/tutorial/theory_to_python/turtlebot_waypoint_block.svg\"\n",
            "      style=\"max-width: 100%;\">\n",
            "</div>\n\n",
            "The waypoint generator produces reference poses $(x_r, y_r, \\\\theta_r)$ for the robot to track. For this tutorial, we'll create a simple square trajectory.\n\n",
            "**State**: Current waypoint index $w_k \\\\in \\\\{0, 1, 2, 3\\\\}$\n\n",
            "**Evolution**:\n\n",
            "$$w_{k+1} = (w_k + 1) \\\\mod 4 \\\\quad \\\\text{when waypoint reached}$$\n\n",
            "**Output**: Reference pose\n\n",
            "$$r_k = \\\\begin{bmatrix} x_r \\\\\\\\ y_r \\\\\\\\ \\\\theta_r \\\\end{bmatrix}$$\n\n",
            "For now, we'll define the waypoints as a simple list. Later, we'll integrate this into a stateful system."
        ]
    })

    # Cell 6: Block 1 - Code
    new_cells.append({
        "cell_type": "code",
        "id": "747cdd23",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from pykal import DynamicalSystem\n",
            "from pykal.data_change import corrupt, prepare  # For realistic sensor noise\n",
            "\n",
            "\n",
            "# Create square trajectory: (x, y, theta) in meters and radians\n",
            "square_waypoints = [\n",
            "    (2.0, 0.0, 0.0),  # Right\n",
            "    (2.0, 2.0, np.pi / 2),  # Up\n",
            "    (0.0, 2.0, np.pi),  # Left\n",
            "    (0.0, 0.0, -np.pi / 2),  # Down (back to start)\n",
            "]\n",
            "\n",
            "\n",
            "# Simple helper function to get waypoint by index\n",
            "def get_waypoint(waypoints, idx):\n",
            "    \"\"\"Get waypoint as numpy array.\"\"\"\n",
            "    x_r, y_r, theta_r = waypoints[idx % len(waypoints)]\n",
            "    return np.array([[x_r], [y_r], [theta_r]])\n",
            "\n",
            "\n",
            "# Test getting waypoints\n",
            "print(\"Waypoint 0:\", get_waypoint(square_waypoints, 0).flatten())\n",
            "print(\"Waypoint 1:\", get_waypoint(square_waypoints, 1).flatten())\n",
            "print(\"Waypoint 2:\", get_waypoint(square_waypoints, 2).flatten())"
        ]
    })

    # Cell 7: Block 2 - Velocity Controller (markdown)
    new_cells.append({
        "cell_type": "markdown",
        "id": "0ebaead8",
        "metadata": {},
        "source": [
            "### Block 2: Velocity Command Generator\n\n",
            "<div style=\"text-align: center;\">\n",
            " <img src=\"../../../_static/tutorial/theory_to_python/turtlebot_controller_block.svg\"\n",
            "      style=\"max-width: 100%;\">\n",
            "</div>\n\n",
            "The velocity controller generates $(v_{cmd}, \\\\omega_{cmd})$ -- linear and angular velocity commands -- to drive the robot toward the reference pose.\n\n",
            "We use a simple proportional controller:\n\n",
            "**Inputs**: Current estimate $\\\\hat{x} = [\\\\hat{x}, \\\\hat{y}, \\\\hat{\\\\theta}]^T$, reference $r = [x_r, y_r, \\\\theta_r]^T$\n\n",
            "**Outputs**:\n\n",
            "$$v_{cmd} = K_v \\\\cdot d \\\\quad \\\\text{(linear velocity proportional to distance)}$$\n\n",
            "$$\\\\omega_{cmd} = K_\\\\omega \\\\cdot (\\\\theta_r - \\\\hat{\\\\theta}) \\\\quad \\\\text{(angular velocity proportional to heading error)}$$\n\n",
            "where $d = \\\\sqrt{(x_r - \\\\hat{x})^2 + (y_r - \\\\hat{y})^2}$\n\n",
            "This controller is **stateless** (no internal dynamics), so we implement it as a pure function:"
        ]
    })

    # Cell 8: Block 2 - Code
    new_cells.append({
        "cell_type": "code",
        "id": "c156c05e",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "def velocity_controller(\n",
            "    xhat: np.ndarray,  # Current estimate [x, y, theta]\n",
            "    r: np.ndarray,  # Reference [x_r, y_r, theta_r]\n",
            "    Kv: float = 0.5,\n",
            "    Komega: float = 1.0,\n",
            "    max_v: float = 0.22,  # TurtleBot max linear velocity (m/s)\n",
            "    max_omega: float = 2.84,  # TurtleBot max angular velocity (rad/s)\n",
            ") -> np.ndarray:\n",
            "    \"\"\"\n",
            "    Proportional controller for TurtleBot velocity commands.\n",
            "\n",
            "    Returns\n",
            "    -------\n",
            "    cmd : np.ndarray\n",
            "        Shape (2,1) containing [v_cmd, omega_cmd]\n",
            "    \"\"\"\n",
            "    # Extract positions\n",
            "    x, y, theta = xhat.flatten()[:3]\n",
            "    x_r, y_r, theta_r = r.flatten()\n",
            "\n",
            "    # Distance to goal\n",
            "    dx = x_r - x\n",
            "    dy = y_r - y\n",
            "    distance = np.sqrt(dx**2 + dy**2)\n",
            "\n",
            "    # Heading error (wrap to [-pi, pi])\n",
            "    heading_error = theta_r - theta\n",
            "    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))\n",
            "\n",
            "    # Proportional control\n",
            "    v_cmd = Kv * distance\n",
            "    omega_cmd = Komega * heading_error\n",
            "\n",
            "    # Saturate commands\n",
            "    v_cmd = np.clip(v_cmd, -max_v, max_v)\n",
            "    omega_cmd = np.clip(omega_cmd, -max_omega, max_omega)\n",
            "\n",
            "    return np.array([[v_cmd], [omega_cmd]])\n",
            "\n",
            "\n",
            "# Test the controller\n",
            "xhat_test = np.array([[0.0], [0.0], [0.0]])\n",
            "r_test = np.array([[1.0], [1.0], [np.pi / 4]])\n",
            "cmd = velocity_controller(xhat_test, r_test)\n",
            "print(f\"Velocity command: v={cmd[0,0]:.3f} m/s, omega={cmd[1,0]:.3f} rad/s\")"
        ]
    })

    # Cell 9: Block 3 - Plant (markdown)
    new_cells.append({
        "cell_type": "markdown",
        "id": "3af5f7bb",
        "metadata": {},
        "source": [
            "### Block 3: TurtleBot (Plant)\n\n",
            "<div style=\"text-align: center;\">\n",
            " <img src=\"../../../_static/tutorial/theory_to_python/turtlebot_plant_block.svg\"\n",
            "      style=\"max-width: 100%;\">\n",
            "</div>\n\n",
            "The TurtleBot is a differential-drive mobile robot. Its motion follows the **unicycle kinematics model**:\n\n",
            "**State**:\n\n",
            "$$x_k = \\\\begin{bmatrix} x \\\\\\\\ y \\\\\\\\ \\\\theta \\\\\\\\ v \\\\\\\\ \\\\omega \\\\end{bmatrix}$$\n\n",
            "where $(x, y)$ is position, $\\\\theta$ is heading, $v$ is linear velocity, $\\\\omega$ is angular velocity.\n\n",
            "**Inputs**: Velocity commands\n\n",
            "$$u_k = \\\\begin{bmatrix} v_{cmd} \\\\\\\\ \\\\omega_{cmd} \\\\end{bmatrix}$$\n\n",
            "**Discrete-time dynamics** (using Euler integration with timestep $\\\\Delta t$):\n\n",
            "$$\n",
            "\\\\begin{aligned}\n",
            "x_{k+1} &= x_k + v_k \\\\cos(\\\\theta_k) \\\\Delta t \\\\\\\\\n",
            "y_{k+1} &= y_k + v_k \\\\sin(\\\\theta_k) \\\\Delta t \\\\\\\\\n",
            "\\\\theta_{k+1} &= \\\\theta_k + \\\\omega_k \\\\Delta t \\\\\\\\\n",
            "v_{k+1} &= v_{cmd} \\\\quad \\\\text{(assume instantaneous velocity response)} \\\\\\\\\n",
            "\\\\omega_{k+1} &= \\\\omega_{cmd}\n",
            "\\\\end{aligned}\n",
            "$$\n\n",
            "We wrap $\\\\theta$ to $[-\\\\pi, \\\\pi]$ to avoid angle wrapping issues.\n\n",
            "**Measurement model**: We observe $[x, y, \\\\theta]$ from wheel odometry (noisy).\n\n",
            "$$y_k = \\\\begin{bmatrix} x_k \\\\\\\\ y_k \\\\\\\\ \\\\theta_k \\\\end{bmatrix} + v_k$$\n\n",
            "where $v_k \\\\sim \\\\mathcal{N}(0, R)$ is measurement noise.\n\n",
            "We implement this discrete-time dynamical system below."
        ]
    })

    # Cell 10: Block 3 - Code
    new_cells.append({
        "cell_type": "code",
        "id": "6ec1b881",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "def turtlebot_f(xk: np.ndarray, uk: np.ndarray, dt: float) -> np.ndarray:\n",
            "    \"\"\"\n",
            "    TurtleBot unicycle dynamics (noise-free).\n",
            "\n",
            "    Parameters\n",
            "    ----------\n",
            "    xk : np.ndarray\n",
            "        Current state [x, y, theta, v, omega], shape (5,1)\n",
            "    uk : np.ndarray\n",
            "        Control input [v_cmd, omega_cmd], shape (2,1)\n",
            "    dt : float\n",
            "        Timestep (seconds)\n",
            "\n",
            "    Returns\n",
            "    -------\n",
            "    xk_next : np.ndarray\n",
            "        Next state, shape (5,1)\n",
            "    \"\"\"\n",
            "    x, y, theta, v, omega = xk.flatten()\n",
            "    v_cmd, omega_cmd = uk.flatten()\n",
            "\n",
            "    # Euler integration\n",
            "    x_new = x + v * np.cos(theta) * dt\n",
            "    y_new = y + v * np.sin(theta) * dt\n",
            "    theta_new = theta + omega * dt\n",
            "\n",
            "    # Wrap theta to [-pi, pi]\n",
            "    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))\n",
            "\n",
            "    # Update velocities (assume instantaneous response)\n",
            "    v_new = v_cmd\n",
            "    omega_new = omega_cmd\n",
            "\n",
            "    return np.array([[x_new], [y_new], [theta_new], [v_new], [omega_new]])\n",
            "\n",
            "\n",
            "def turtlebot_h(xk: np.ndarray) -> np.ndarray:\n",
            "    \"\"\"\n",
            "    Measurement function: observe [x, y, theta] from odometry.\n",
            "\n",
            "    Parameters\n",
            "    ----------\n",
            "    xk : np.ndarray\n",
            "        State [x, y, theta, v, omega], shape (5,1)\n",
            "\n",
            "    Returns\n",
            "    -------\n",
            "    yk : np.ndarray\n",
            "        Measurement [x, y, theta], shape (3,1)\n",
            "    \"\"\"\n",
            "    return xk[:3, :]\n",
            "\n",
            "\n",
            "# Create the plant DynamicalSystem\n",
            "plant = DynamicalSystem(f=turtlebot_f, h=turtlebot_h, state_name=\"xk\")\n",
            "\n",
            "# Test the plant\n",
            "xk_test = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])  # Start at origin, stationary\n",
            "uk_test = np.array([[0.2], [0.5]])  # Move forward and turn\n",
            "dt = 0.1\n",
            "\n",
            "xk_next, yk = plant.step(\n",
            "    param_dict={\"xk\": xk_test, \"uk\": uk_test, \"dt\": dt}, return_state=True\n",
            ")\n",
            "print(\"After one step:\")\n",
            "print(f\"  State: {xk_next.flatten()}\")\n",
            "print(f\"  Measurement: {yk.flatten()}\")"
        ]
    })

    # Cell 11: Block 4 - Observer (markdown)
    new_cells.append({
        "cell_type": "markdown",
        "id": "424cf9ac",
        "metadata": {},
        "source": [
            "### Block 4: Kalman Filter (Observer)\n\n",
            "<div style=\"text-align: center;\">\n",
            " <img src=\"../../../_static/tutorial/theory_to_python/turtlebot_observer_block.svg\"\n",
            "      style=\"max-width: 100%;\">\n",
            "</div>\n\n",
            "The Kalman filter estimates the TurtleBot's state $\\\\hat{x}_k$ by fusing:\n",
            "1. **Prediction**: Motion model propagates state forward using control inputs\n",
            "2. **Update**: Noisy odometry measurements correct the prediction\n\n",
            "Since the dynamics are **nonlinear** (due to $\\\\cos(\\\\theta)$ and $\\\\sin(\\\\theta)$), we use the **Extended Kalman Filter (EKF)**, which linearizes around the current estimate.\n\n",
            "**Jacobian of dynamics** (for covariance propagation):\n\n",
            "$$\n",
            "F_k = \\\\frac{\\\\partial f}{\\\\partial x}\\\\bigg|_{\\\\hat{x}_k} =\n",
            "\\\\begin{bmatrix}\n",
            "1 & 0 & -v \\\\sin(\\\\theta) \\\\Delta t & \\\\cos(\\\\theta) \\\\Delta t & 0 \\\\\\\\\n",
            "0 & 1 & v \\\\cos(\\\\theta) \\\\Delta t & \\\\sin(\\\\theta) \\\\Delta t & 0 \\\\\\\\\n",
            "0 & 0 & 1 & 0 & \\\\Delta t \\\\\\\\\n",
            "0 & 0 & 0 & 1 & 0 \\\\\\\\\n",
            "0 & 0 & 0 & 0 & 1\n",
            "\\\\end{bmatrix}\n",
            "$$\n\n",
            "**Jacobian of measurement**:\n\n",
            "$$\n",
            "H_k = \\\\begin{bmatrix}\n",
            "1 & 0 & 0 & 0 & 0 \\\\\\\\\n",
            "0 & 1 & 0 & 0 & 0 \\\\\\\\\n",
            "0 & 0 & 1 & 0 & 0\n",
            "\\\\end{bmatrix}\n",
            "$$\n\n",
            "**Noise covariances**:\n",
            "- Process noise $Q$: Uncertainty in velocity model\n",
            "- Measurement noise $R$: Wheel odometry error\n\n",
            "The implementation can be found in [KF](../../../src/pykal/algorithm_library/estimators/kf.py)."
        ]
    })

    # Cell 12: Block 4 - Code
    new_cells.append({
        "cell_type": "code",
        "id": "d85e0876",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "from pykal.algorithm_library.estimators import kf\n",
            "\n",
            "\n",
            "def compute_F_jacobian(xhat: np.ndarray, dt: float) -> np.ndarray:\n",
            "    \"\"\"\n",
            "    Compute Jacobian of dynamics for TurtleBot.\n",
            "\n",
            "    Parameters\n",
            "    ----------\n",
            "    xhat : np.ndarray\n",
            "        Current state estimate [x, y, theta, v, omega], shape (5,1)\n",
            "    dt : float\n",
            "        Timestep\n",
            "\n",
            "    Returns\n",
            "    -------\n",
            "    F : np.ndarray\n",
            "        Jacobian matrix, shape (5, 5)\n",
            "    \"\"\"\n",
            "    _, _, theta, v, _ = xhat.flatten()\n",
            "\n",
            "    F = np.array(\n",
            "        [\n",
            "            [1, 0, -v * np.sin(theta) * dt, np.cos(theta) * dt, 0],\n",
            "            [0, 1, v * np.cos(theta) * dt, np.sin(theta) * dt, 0],\n",
            "            [0, 0, 1, 0, dt],\n",
            "            [0, 0, 0, 1, 0],\n",
            "            [0, 0, 0, 0, 1],\n",
            "        ]\n",
            "    )\n",
            "\n",
            "    return F\n",
            "\n",
            "\n",
            "def compute_H_jacobian() -> np.ndarray:\n",
            "    \"\"\"\n",
            "    Compute Jacobian of measurement function (constant for this system).\n",
            "\n",
            "    Returns\n",
            "    -------\n",
            "    H : np.ndarray\n",
            "        Jacobian matrix, shape (3, 5)\n",
            "    \"\"\"\n",
            "    return np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])\n",
            "\n",
            "\n",
            "# Process and measurement noise covariances\n",
            "Q_turtlebot = np.diag([0.01, 0.01, 0.02, 0.1, 0.1])  # Process noise\n",
            "R_turtlebot = np.diag([0.05, 0.05, 0.1])  # Odometry noise (5cm, 5cm, ~6 degrees)\n",
            "\n",
            "# Create the observer DynamicalSystem\n",
            "observer = DynamicalSystem(f=kf.f, h=kf.h, state_name=\"xhat_P\")\n",
            "\n",
            "# Test the observer (single step)\n",
            "xhat_0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])\n",
            "P_0 = np.diag([0.1, 0.1, 0.1, 1.0, 1.0])\n",
            "xhat_P_test = (xhat_0, P_0)\n",
            "\n",
            "# Simulate receiving a noisy measurement\n",
            "yk_noisy = np.array([[0.02], [0.01], [0.05]])  # Noisy measurement\n",
            "\n",
            "xhat_P_new, xhat_out = observer.step(\n",
            "    param_dict={\n",
            "        \"xhat_P\": xhat_P_test,\n",
            "        \"yk\": yk_noisy,\n",
            "        \"f\": turtlebot_f,\n",
            "        \"f_params\": {\"xk\": xhat_0, \"uk\": np.array([[0.0], [0.0]]), \"dt\": 0.1},\n",
            "        \"h\": turtlebot_h,\n",
            "        \"h_params\": {\"xk\": xhat_0},\n",
            "        \"Fk\": compute_F_jacobian(xhat_0, dt=0.1),\n",
            "        \"Qk\": Q_turtlebot,\n",
            "        \"Hk\": compute_H_jacobian(),\n",
            "        \"Rk\": R_turtlebot,\n",
            "    },\n",
            "    return_state=True,\n",
            ")\n",
            "\n",
            "print(\"Observer estimate after one measurement:\")\n",
            "print(xhat_out.flatten())"
        ]
    })

    # Cell 13: Simulation header
    new_cells.append({
        "cell_type": "markdown",
        "id": "af717eb1",
        "metadata": {},
        "source": [
            "## Simulation\n",
            "\n",
            "![Complete TurtleBot system](../../../_static/tutorial/theory_to_python/turtlebot_composition_of_dynamical_systems.svg)\n\n",
            "We now simulate the complete closed-loop system, integrating all four dynamical components:\n",
            "1. **Waypoint Generator** → reference pose $r_k$\n",
            "2. **Velocity Controller** → velocity commands $u_k$ (using $r_k$ and $\\\\hat{x}_k$)\n",
            "3. **TurtleBot Plant** → true state evolution and noisy measurements $y_k$\n",
            "4. **Kalman Filter** → state estimate $\\\\hat{x}_k$ (using $u_k$ and $y_k$)"
        ]
    })

    # Cell 14: System Parameters
    new_cells.append({
        "cell_type": "markdown",
        "id": "fe5bcae6",
        "metadata": {},
        "source": [
            "### System Parameters"
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "id": "e235120d",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Time parameters\n",
            "dt = 0.1  # Sampling time (seconds)\n",
            "switch_time = 15.0  # Time at each waypoint (seconds)\n",
            "\n",
            "# Controller gains\n",
            "Kv = 0.5  # Linear velocity gain\n",
            "Komega = 1.5  # Angular velocity gain\n",
            "\n",
            "# Kalman filter parameters\n",
            "Q = np.diag([0.01, 0.01, 0.02, 0.1, 0.1])  # Process noise covariance\n",
            "R = np.diag([0.05, 0.05, 0.1])  # Measurement noise covariance"
        ]
    })

    # Cell 15: Initial Conditions
    new_cells.append({
        "cell_type": "markdown",
        "id": "8ecb57fc",
        "metadata": {},
        "source": [
            "### Initial Conditions"
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "id": "713453e9",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Initial states\n",
            "waypoint_idx = 0  # Start at first waypoint\n",
            "time_at_waypoint = 0.0  # Time spent at current waypoint\n",
            "\n",
            "xk = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])  # Plant state: start at origin\n",
            "xhat = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])  # Observer estimate\n",
            "P = np.diag([0.1, 0.1, 0.1, 1.0, 1.0])  # Covariance matrix\n",
            "xhat_P = (xhat, P)  # Observer state tuple\n",
            "\n",
            "# Storage for plotting\n",
            "time_hist = []\n",
            "reference_hist = []\n",
            "true_state_hist = []\n",
            "estimate_hist = []\n",
            "measurement_hist = []\n",
            "command_hist = []\n",
            "error_hist = []"
        ]
    })

    # Cell 16: Simulate
    new_cells.append({
        "cell_type": "markdown",
        "id": "c41b7b08",
        "metadata": {},
        "source": [
            "### Simulate"
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "id": "5b82f5f8",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Simulation time\n",
            "T_sim = 60.0  # seconds\n",
            "time_steps = np.arange(0, T_sim, dt)\n",
            "\n",
            "# Run closed-loop simulation\n",
            "for tk in time_steps:\n",
            "    # 1. Waypoint generator (simple switching logic)\n",
            "    time_at_waypoint += dt\n",
            "    if time_at_waypoint >= switch_time:\n",
            "        waypoint_idx = (waypoint_idx + 1) % len(square_waypoints)\n",
            "        time_at_waypoint = 0.0\n",
            "    \n",
            "    rk = get_waypoint(square_waypoints, waypoint_idx)\n",
            "    \n",
            "    # 2. Extract current state estimate from observer\n",
            "    xhat = observer.h(xhat_P)\n",
            "    \n",
            "    # 3. Velocity controller step\n",
            "    uk = velocity_controller(xhat, rk, Kv=Kv, Komega=Komega)\n",
            "    \n",
            "    # 4. Plant step (true dynamics)\n",
            "    xk, yk_clean = plant.step(\n",
            "        return_state=True,\n",
            "        param_dict={\"xk\": xk, \"uk\": uk, \"dt\": dt}\n",
            "    )\n",
            "    \n",
            "    # 5. Add measurement noise (realistic corruption)\n",
            "    yk_flat = yk_clean.flatten()\n",
            "    \n",
            "    # X: bias (wheel diameter mismatch) + Gaussian noise\n",
            "    yk_corrupted_x = corrupt.with_bias(yk_flat[0], bias=0.02)\n",
            "    yk_corrupted_x = yk_corrupted_x + np.random.normal(0, np.sqrt(R[0, 0]))\n",
            "    \n",
            "    # Y: Gaussian noise\n",
            "    yk_corrupted_y = yk_flat[1] + np.random.normal(0, np.sqrt(R[1, 1]))\n",
            "    \n",
            "    # Theta: Gaussian noise + occasional spikes (bumps)\n",
            "    yk_corrupted_theta = yk_flat[2] + np.random.normal(0, np.sqrt(R[2, 2]))\n",
            "    if np.random.rand() < 0.02:  # 2% spike rate\n",
            "        yk_corrupted_theta += np.random.choice([-1, 1]) * 0.3\n",
            "    \n",
            "    yk_noisy = np.array([[yk_corrupted_x], [yk_corrupted_y], [yk_corrupted_theta]])\n",
            "    \n",
            "    # 6. Observer step (EKF)\n",
            "    Fk = compute_F_jacobian(xhat, dt)\n",
            "    Hk = compute_H_jacobian()\n",
            "    \n",
            "    xhat_P, xhat_obs = observer.step(\n",
            "        return_state=True,\n",
            "        param_dict={\n",
            "            \"xhat_P\": xhat_P,\n",
            "            \"yk\": yk_noisy,\n",
            "            \"f\": turtlebot_f,\n",
            "            \"f_params\": {\"xk\": xhat, \"uk\": uk, \"dt\": dt},\n",
            "            \"h\": turtlebot_h,\n",
            "            \"h_params\": {\"xk\": xhat},\n",
            "            \"Fk\": Fk,\n",
            "            \"Qk\": Q,\n",
            "            \"Hk\": Hk,\n",
            "            \"Rk\": R,\n",
            "        },\n",
            "    )\n",
            "    \n",
            "    # Store results\n",
            "    time_hist.append(tk)\n",
            "    reference_hist.append(rk.flatten())\n",
            "    true_state_hist.append(xk.flatten())\n",
            "    estimate_hist.append(xhat.flatten())\n",
            "    measurement_hist.append(yk_noisy.flatten())\n",
            "    command_hist.append(uk.flatten())\n",
            "    error_hist.append((xk - xhat).flatten())\n",
            "\n",
            "# Convert to numpy arrays\n",
            "reference_hist = np.array(reference_hist)\n",
            "true_state_hist = np.array(true_state_hist)\n",
            "estimate_hist = np.array(estimate_hist)\n",
            "measurement_hist = np.array(measurement_hist)\n",
            "command_hist = np.array(command_hist)\n",
            "error_hist = np.array(error_hist)"
        ]
    })

    # Cell 17: Visualize
    new_cells.append({
        "cell_type": "markdown",
        "id": "5d8d58e8",
        "metadata": {},
        "source": [
            "### Visualize\n",
            "\n",
            "We can visualize the pertinent values of our system to assure correct behavior. Note how the Kalman filter smooths out the noisy measurements and provides accurate state estimates even in the presence of sensor corruption."
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "id": "e527e920",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "fig, axs = plt.subplots(3, 2, figsize=(14, 12))\n",
            "\n",
            "# Plot 1: 2D Trajectory\n",
            "ax = axs[0, 0]\n",
            "ax.plot(\n",
            "    true_state_hist[:, 0],\n",
            "    true_state_hist[:, 1],\n",
            "    \"b-\",\n",
            "    label=\"True Trajectory\",\n",
            "    linewidth=2,\n",
            "    alpha=0.7,\n",
            ")\n",
            "ax.plot(\n",
            "    estimate_hist[:, 0],\n",
            "    estimate_hist[:, 1],\n",
            "    \"r--\",\n",
            "    label=\"Estimated Trajectory\",\n",
            "    linewidth=2,\n",
            "    alpha=0.7,\n",
            ")\n",
            "ax.scatter(\n",
            "    reference_hist[:, 0],\n",
            "    reference_hist[:, 1],\n",
            "    c=\"green\",\n",
            "    s=100,\n",
            "    marker=\"*\",\n",
            "    label=\"Waypoints\",\n",
            "    zorder=5,\n",
            ")\n",
            "ax.set_xlabel(\"X Position (m)\", fontsize=11)\n",
            "ax.set_ylabel(\"Y Position (m)\", fontsize=11)\n",
            "ax.set_title(\"2D Trajectory Tracking\", fontsize=13, fontweight=\"bold\")\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "ax.axis(\"equal\")\n",
            "\n",
            "# Plot 2: X Position vs Time\n",
            "ax = axs[0, 1]\n",
            "ax.plot(time_hist, true_state_hist[:, 0], \"b-\", label=\"True X\", alpha=0.7)\n",
            "ax.plot(\n",
            "    time_hist, estimate_hist[:, 0], \"r--\", label=\"Estimated X\", alpha=0.7\n",
            ")\n",
            "ax.scatter(\n",
            "    time_hist[::10],\n",
            "    measurement_hist[::10, 0],\n",
            "    c=\"gray\",\n",
            "    s=10,\n",
            "    alpha=0.3,\n",
            "    label=\"Measurements\",\n",
            ")\n",
            "ax.set_ylabel(\"X Position (m)\", fontsize=11)\n",
            "ax.set_title(\"X Position Over Time\", fontsize=13, fontweight=\"bold\")\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# Plot 3: Y Position vs Time\n",
            "ax = axs[1, 0]\n",
            "ax.plot(time_hist, true_state_hist[:, 1], \"b-\", label=\"True Y\", alpha=0.7)\n",
            "ax.plot(\n",
            "    time_hist, estimate_hist[:, 1], \"r--\", label=\"Estimated Y\", alpha=0.7\n",
            ")\n",
            "ax.scatter(\n",
            "    time_hist[::10],\n",
            "    measurement_hist[::10, 1],\n",
            "    c=\"gray\",\n",
            "    s=10,\n",
            "    alpha=0.3,\n",
            "    label=\"Measurements\",\n",
            ")\n",
            "ax.set_ylabel(\"Y Position (m)\", fontsize=11)\n",
            "ax.set_title(\"Y Position Over Time\", fontsize=13, fontweight=\"bold\")\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# Plot 4: Heading vs Time\n",
            "ax = axs[1, 1]\n",
            "ax.plot(\n",
            "    time_hist,\n",
            "    np.rad2deg(true_state_hist[:, 2]),\n",
            "    \"b-\",\n",
            "    label=\"True Heading\",\n",
            "    alpha=0.7,\n",
            ")\n",
            "ax.plot(\n",
            "    time_hist,\n",
            "    np.rad2deg(estimate_hist[:, 2]),\n",
            "    \"r--\",\n",
            "    label=\"Estimated Heading\",\n",
            "    alpha=0.7,\n",
            ")\n",
            "ax.set_ylabel(\"Heading (degrees)\", fontsize=11)\n",
            "ax.set_title(\"Heading Over Time\", fontsize=13, fontweight=\"bold\")\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# Plot 5: Velocity Commands\n",
            "ax = axs[2, 0]\n",
            "ax.plot(\n",
            "    time_hist,\n",
            "    command_hist[:, 0],\n",
            "    \"g-\",\n",
            "    label=\"Linear Velocity\",\n",
            "    linewidth=1.5,\n",
            ")\n",
            "ax.set_ylabel(\"Linear Velocity (m/s)\", fontsize=11)\n",
            "ax.set_xlabel(\"Time (s)\", fontsize=11)\n",
            "ax.set_title(\"Control Commands\", fontsize=13, fontweight=\"bold\")\n",
            "ax.legend(loc=\"upper left\")\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "ax_omega = ax.twinx()\n",
            "ax_omega.plot(\n",
            "    time_hist,\n",
            "    command_hist[:, 1],\n",
            "    \"purple\",\n",
            "    label=\"Angular Velocity\",\n",
            "    linewidth=1.5,\n",
            "    alpha=0.7,\n",
            ")\n",
            "ax_omega.set_ylabel(\"Angular Velocity (rad/s)\", fontsize=11, color=\"purple\")\n",
            "ax_omega.tick_params(axis=\"y\", labelcolor=\"purple\")\n",
            "ax_omega.legend(loc=\"upper right\")\n",
            "\n",
            "# Plot 6: Position Estimation Error\n",
            "ax = axs[2, 1]\n",
            "position_error = np.sqrt(\n",
            "    error_hist[:, 0] ** 2 + error_hist[:, 1] ** 2\n",
            ")\n",
            "ax.plot(time_hist, position_error, \"m-\", label=\"Position Error\", linewidth=1.5)\n",
            "ax.set_xlabel(\"Time (s)\", fontsize=11)\n",
            "ax.set_ylabel(\"Position Error (m)\", fontsize=11)\n",
            "ax.set_title(\"Estimation Error Magnitude\", fontsize=13, fontweight=\"bold\")\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f\"Mean position error: {np.mean(position_error):.4f} m\")\n",
            "print(f\"Max position error: {np.max(position_error):.4f} m\")"
        ]
    })

    # Cell 18: Callback wrapper header
    new_cells.append({
        "cell_type": "markdown",
        "id": "83b2401b",
        "metadata": {},
        "source": [
            "## Callback Wrapper\n\n",
            "Just as we did with the cruise control system, we can wrap the **entire TurtleBot navigation system** in a single callback. This encapsulates all four dynamical systems (waypoint generator, velocity controller, plant, and observer) and their states, making it easy to reuse and experiment with different configurations. This pattern is particularly useful for integration with ROS2, as we'll see in later tutorials."
        ]
    })

    # Keep the existing callback implementation and experimentation cells from the original
    # Find them in the original notebook
    with open(nb_path, 'r') as f:
        original_nb = json.load(f)

    # Find the callback cells (they're near the end)
    callback_cells_started = False
    for cell in original_nb['cells']:
        if cell['cell_type'] == 'code' and 'def initialize_waypoint_generator' in ''.join(cell.get('source', [])):
            callback_cells_started = True

        if callback_cells_started:
            # Keep these cells but update IDs
            if cell['cell_type'] == 'code' and 'def initialize_waypoint_generator' in ''.join(cell.get('source', [])):
                new_cells.append({
                    "cell_type": "code",
                    "id": "fee3d796",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": cell['source']
                })
            elif cell['cell_type'] == 'markdown' and 'Experimentation' in ''.join(cell.get('source', [])):
                new_cells.append({
                    "cell_type": "markdown",
                    "id": "723ac54b",
                    "metadata": {},
                    "source": cell['source']
                })
                break

    # Add experimentation section
    new_cells.append({
        "cell_type": "markdown",
        "id": "5a087b7d",
        "metadata": {},
        "source": [
            "## Experimentation\n\n",
            "Now that we have a working system, try experimenting with different parameters:\n\n",
            "**Noise tuning**:\n",
            "- Increase measurement noise `R` to simulate worse odometry (wheel slip, rough terrain)\n",
            "- Increase process noise `Q` to account for model uncertainty\n",
            "- Observe how the filter trades off model prediction vs measurements\n\n",
            "**Controller tuning**:\n",
            "- Adjust `Kv` and `Komega` to change aggressiveness\n",
            "- Try different waypoint patterns (figure-eight, circle, random)\n",
            "- Add velocity limits to simulate real robot constraints\n\n",
            "**Observer comparison**:\n",
            "- Run simulation with KF disabled (use raw measurements)\n",
            "- Compare estimation error with and without filtering\n",
            "- Visualize covariance ellipses over time\n\n",
            "**Challenge**: Can you tune Q and R to minimize position error while maintaining smooth estimates?"
        ]
    })

    # Final cell: Parent link
    new_cells.append({
        "cell_type": "markdown",
        "id": "555c071f",
        "metadata": {},
        "source": [
            "[← Control Systems as Dynamical Systems](../../../getting_started/theory_to_python/control_systems_as_dynamical_systems.rst)"
        ]
    })

    # Write the restructured notebook
    nb['cells'] = new_cells

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully restructured {nb_path}")
    print(f"Total cells: {len(new_cells)}")


if __name__ == "__main__":
    restructure_turtlebot_notebook()
