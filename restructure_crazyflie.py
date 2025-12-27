#!/usr/bin/env python3
"""Restructure crazyflie notebook to match turtlebot/cruise_control pedagogical approach."""

import json

nb_path = "docs/source/notebooks/tutorial/theory_to_python/crazyflie_sensor_fusion.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Build new cell structure
new_cells = []

# Cell 0: Parent link
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-0",
    "metadata": {},
    "source": [
        "[← Control Systems as Dynamical Systems](../../../getting_started/theory_to_python/control_systems_as_dynamical_systems.rst)\n"
    ]
})

# Cell 1: Title
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-1",
    "metadata": {},
    "source": [
        "# Example: Crazyflie Multi-Sensor Fusion\n"
    ]
})

# Cell 2: Intro paragraph + System Overview
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-2",
    "metadata": {},
    "source": [
        "Suppose we want to stabilize a quadrotor in 3D space. We need accurate position and velocity estimates to maintain stable flight. However, our sensors have complementary strengths and weaknesses: motion capture is precise but slow, the barometer drifts with temperature, and the IMU is noisy but fast.\n",
        "\n",
        "In this notebook, we extend our state estimation framework to handle **multiple asynchronous sensors** with different characteristics. The Crazyflie quadrotor fuses motion capture, barometer, and IMU data using a Kalman filter that optimally weights each sensor based on its noise properties. This demonstrates how sensor fusion can achieve better performance than any single sensor alone.\n",
        "\n",
        "## System Overview\n",
        "\n",
        "We model the quadrotor dynamics, implement sensor-specific noise characteristics, design the multi-sensor observer, and integrate everything into a complete feedback system. This notebook bridges theory and practice for aerial robot state estimation with heterogeneous sensors.\n"
    ]
})

# Cell 3: Block Diagram
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-3",
    "metadata": {},
    "source": [
        "## Block Diagram\n",
        "\n",
        "The Crazyflie system has multiple sensor streams feeding a central estimator:\n",
        "\n",
        "![Crazyflie multi-sensor fusion system](../../../_static/tutorial/theory_to_python/crazyflie_multisensor_system.svg)\n",
        "\n",
        "where **setpoint generator** produces reference positions (e.g., hover at [0, 0, 1]), **position controller** outputs velocity commands to track the reference, **Crazyflie** is our quadrotor plant with 3D dynamics, **motion capture** is a 3D position sensor (10 Hz, 5mm accuracy), **barometer** measures altitude (20 Hz, 10cm accuracy, subject to drift), **IMU** measures velocity from accelerometer integration (100 Hz, noisy), and **Kalman Filter** is the state observer that fuses all three sensors optimally.\n",
        "\n",
        ":::{note}\n",
        "Different sensors operate at different rates and measure different aspects of the state. The KF intelligently weights each sensor based on its noise characteristics. For pedagogical clarity, we'll run all sensors synchronously in this notebook.\n",
        ":::\n"
    ]
})

# Cell 4: Discrete-time Dynamical Systems
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-4",
    "metadata": {},
    "source": [
        "### Discrete-time Dynamical Systems\n",
        "\n",
        "We cast each component as a discrete-time dynamical system:\n",
        "\n",
        "![Crazyflie as composition of dynamical systems](../../../_static/tutorial/theory_to_python/crazyflie_composition_of_dynamical_systems.svg)\n",
        "\n",
        "We discuss the derivation of each dynamical system block below.\n",
        "\n",
        ":::{note}\n",
        "Notice the **three sensor blocks** all feeding into a single observer. The KF combines their measurements by concatenating them into a single measurement vector. Unlike the TurtleBot's nonlinear dynamics, the Crazyflie uses a linear constant-velocity model, making the KF exact (no linearization needed).\n",
        ":::\n"
    ]
})

# Cell 5: Block 1 - Setpoint Generator (markdown)
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-5",
    "metadata": {},
    "source": [
        "### Block 1: Setpoint Generator\n",
        "\n",
        "<div style=\"text-align: center;\">\n",
        " <img src=\"../../../_static/tutorial/theory_to_python/crazyflie_setpoint_block.svg\"\n",
        "      style=\"max-width: 100%;\">\n",
        "</div>\n",
        "\n",
        "For this tutorial, we'll command the drone to hover at a fixed position, then transition to new positions after some time.\n",
        "\n",
        "**State**: Target position $s_k = [x_r, y_r, z_r]^T$\n",
        "\n",
        "**Evolution**: Step function that switches between waypoints at specified times\n",
        "\n",
        "**Output**: Reference position\n",
        "\n",
        "$$\n",
        "r_k = s_k\n",
        "$$\n",
        "\n",
        "For now, we'll define waypoints as a simple list. Later, we'll integrate this into a stateful system.\n"
    ]
})

# Cell 6: Block 1 - Code
new_cells.append({
    "cell_type": "code",
    "id": "cell-6",
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
        "# Create hover trajectory: positions in meters\n",
        "hover_waypoints = [\n",
        "    (0.0, 0.0, 1.0),   # Start: hover at 1m altitude\n",
        "    (1.0, 1.0, 1.5),   # Move to (1, 1, 1.5)\n",
        "    (0.0, 0.0, 1.0),   # Return to start\n",
        "]\n",
        "\n",
        "\n",
        "# Simple helper function to get waypoint by index\n",
        "def get_hover_waypoint(waypoints, idx):\n",
        "    \"\"\"Get waypoint as numpy array.\"\"\"\n",
        "    x_r, y_r, z_r = waypoints[idx % len(waypoints)]\n",
        "    return np.array([[x_r], [y_r], [z_r]])\n",
        "\n",
        "\n",
        "# Test getting waypoints\n",
        "print(\"Waypoint 0:\", get_hover_waypoint(hover_waypoints, 0).flatten())\n",
        "print(\"Waypoint 1:\", get_hover_waypoint(hover_waypoints, 1).flatten())\n",
        "print(\"Waypoint 2:\", get_hover_waypoint(hover_waypoints, 2).flatten())\n"
    ]
})

# Cell 7: Block 2 - Controller (markdown)
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-7",
    "metadata": {},
    "source": [
        "### Block 2: Position Controller\n",
        "\n",
        "<div style=\"text-align: center;\">\n",
        " <img src=\"../../../_static/tutorial/theory_to_python/crazyflie_controller_block.svg\"\n",
        "      style=\"max-width: 100%;\">\n",
        "</div>\n",
        "\n",
        "The position controller generates velocity commands $[v_x, v_y, v_z]$ to drive the drone toward the reference position.\n",
        "\n",
        "**Inputs**: Reference position $r = [x_r, y_r, z_r]^T$, position estimate $\\hat{p} = [\\hat{x}, \\hat{y}, \\hat{z}]^T$\n",
        "\n",
        "**Outputs**: Velocity command\n",
        "\n",
        "$$\n",
        "\\vec{v}_{cmd} = K_p (\\vec{r} - \\hat{\\vec{p}})\n",
        "$$\n",
        "\n",
        "where $K_p$ is the proportional gain.\n",
        "\n",
        "This controller is **stateless** (no internal dynamics), so we implement it as a pure function:\n"
    ]
})

# Cell 8: Block 2 - Code
new_cells.append({
    "cell_type": "code",
    "id": "cell-8",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "def position_controller(\n",
        "    phat: np.ndarray,  # Estimated position [x, y, z]\n",
        "    r: np.ndarray,      # Reference position [x_r, y_r, z_r]\n",
        "    Kp: float = 1.0,\n",
        "    max_vel: float = 0.5  # m/s\n",
        ") -> np.ndarray:\n",
        "    \"\"\"\n",
        "    Proportional position controller for Crazyflie.\n",
        "    \n",
        "    Returns\n",
        "    -------\n",
        "    v_cmd : np.ndarray\n",
        "        Velocity command [vx, vy, vz], shape (3,1)\n",
        "    \"\"\"\n",
        "    # Position error\n",
        "    error = r - phat\n",
        "    \n",
        "    # Proportional control\n",
        "    v_cmd = Kp * error\n",
        "    \n",
        "    # Saturate\n",
        "    v_cmd = np.clip(v_cmd, -max_vel, max_vel)\n",
        "    \n",
        "    return v_cmd\n",
        "\n",
        "\n",
        "# Test the controller\n",
        "phat_test = np.array([[0.0], [0.0], [0.8]])\n",
        "r_test = np.array([[1.0], [0.5], [1.0]])\n",
        "v_cmd = position_controller(phat_test, r_test, Kp=0.8)\n",
        "print(f\"Velocity command: {v_cmd.flatten()}\")\n"
    ]
})

# Cells 9-13: Keep blocks 3-5 as they are (plant, multi-sensor, observer)
# Copy from original notebook
for i in range(8, 14):  # cells 8-13 in original are blocks 3-5
    if i < len(nb['cells']):
        new_cells.append(nb['cells'][i])

# Now add the Simulation section (direct composition)
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-sim-header",
    "metadata": {},
    "source": [
        "## Simulation\n",
        "\n",
        "![Complete Crazyflie system](../../../_static/tutorial/theory_to_python/crazyflie_composition_of_dynamical_systems.svg)\n",
        "\n",
        "We now simulate the complete closed-loop system, integrating all five dynamical components:\n",
        "1. **Setpoint Generator** → reference position $r_k$\n",
        "2. **Position Controller** → velocity commands $u_k$ (using $r_k$ and $\\hat{p}_k$)\n",
        "3. **Crazyflie Plant** → true state evolution\n",
        "4. **Multi-Sensor Array** → measurements $y_k$ from mocap, baro, IMU\n",
        "5. **Kalman Filter** → fused state estimate $\\hat{x}_k$ (using $u_k$ and $y_k$)\n"
    ]
})

# System Parameters
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-sim-params-header",
    "metadata": {},
    "source": [
        "### System Parameters\n"
    ]
})

new_cells.append({
    "cell_type": "code",
    "id": "cell-sim-params",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Time parameters\n",
        "dt = 0.01  # Sampling time (seconds) - 100 Hz\n",
        "switch_time = 20.0  # Time at each waypoint (seconds)\n",
        "\n",
        "# Controller gains\n",
        "Kp = 0.8  # Position control gain\n",
        "\n",
        "# Kalman filter parameters\n",
        "Q = np.diag([0.001, 0.001, 0.001, 0.1, 0.1, 0.1])  # Process noise\n",
        "R_mocap = np.diag([0.005, 0.005, 0.005])  # Motion capture noise (5mm)\n",
        "R_baro = np.array([[0.1]])  # Barometer noise (10cm)\n",
        "R_imu = np.diag([0.1, 0.1, 0.1])  # IMU noise (0.1 m/s)\n"
    ]
})

# Initial Conditions
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-sim-init-header",
    "metadata": {},
    "source": [
        "### Initial Conditions\n"
    ]
})

new_cells.append({
    "cell_type": "code",
    "id": "cell-sim-init",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Initial states\n",
        "waypoint_idx = 0  # Start at first waypoint\n",
        "time_at_waypoint = 0.0  # Time spent at current waypoint\n",
        "\n",
        "xk = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])  # Plant state: hover at 1m\n",
        "xhat = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])  # Observer estimate\n",
        "P = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])  # Covariance matrix\n",
        "xhat_P = (xhat, P)  # Observer state tuple\n",
        "\n",
        "# Storage for plotting\n",
        "time_hist = []\n",
        "reference_hist = []\n",
        "true_state_hist = []\n",
        "estimate_hist = []\n",
        "measurement_mocap_hist = []\n",
        "measurement_baro_hist = []\n",
        "command_hist = []\n",
        "error_hist = []\n"
    ]
})

# Simulate
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-sim-run-header",
    "metadata": {},
    "source": [
        "### Simulate\n"
    ]
})

new_cells.append({
    "cell_type": "code",
    "id": "cell-sim-run",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Simulation time\n",
        "T_sim = 60.0  # seconds\n",
        "time_steps = np.arange(0, T_sim, dt)\n",
        "\n",
        "# Import KF from algorithm library\n",
        "from pykal.algorithm_library.estimators import kf as KF_module\n",
        "\n",
        "# Create observer DynamicalSystem\n",
        "observer = DynamicalSystem(f=KF_module.f, h=KF_module.h, state_name=\"xhat_P\")\n",
        "\n",
        "# Run closed-loop simulation\n",
        "for tk in time_steps:\n",
        "    # 1. Setpoint generator (simple switching logic)\n",
        "    time_at_waypoint += dt\n",
        "    if time_at_waypoint >= switch_time:\n",
        "        waypoint_idx = (waypoint_idx + 1) % len(hover_waypoints)\n",
        "        time_at_waypoint = 0.0\n",
        "    \n",
        "    rk = get_hover_waypoint(hover_waypoints, waypoint_idx)\n",
        "    \n",
        "    # 2. Extract position estimate from observer\n",
        "    xhat = observer.h(xhat_P)\n",
        "    phat = xhat[:3]\n",
        "    \n",
        "    # 3. Position controller step\n",
        "    uk = position_controller(phat, rk, Kp=Kp)\n",
        "    \n",
        "    # 4. Plant step (true dynamics)\n",
        "    xk, _ = crazyflie_plant.step(\n",
        "        return_state=True,\n",
        "        param_dict={\"xk\": xk, \"uk\": uk, \"dt\": dt}\n",
        "    )\n",
        "    \n",
        "    # 5. Generate multi-sensor measurements\n",
        "    yk_combined, R_combined = generate_multisensor_measurement(\n",
        "        xk, R_mocap, R_baro, R_imu, use_realistic_corruption=True\n",
        "    )\n",
        "    \n",
        "    # 6. Observer step (KF)\n",
        "    Fk = compute_F_crazyflie(dt)\n",
        "    Hk = compute_H_multisensor()\n",
        "    \n",
        "    xhat_P, xhat_obs = observer.step(\n",
        "        return_state=True,\n",
        "        param_dict={\n",
        "            \"xhat_P\": xhat_P,\n",
        "            \"yk\": yk_combined,\n",
        "            \"f\": crazyflie_f,\n",
        "            \"f_params\": {\"xk\": xhat, \"uk\": uk, \"dt\": dt},\n",
        "            \"h\": h_multisensor,\n",
        "            \"h_params\": {\"xk\": xhat},\n",
        "            \"Fk\": Fk,\n",
        "            \"Qk\": Q,\n",
        "            \"Hk\": Hk,\n",
        "            \"Rk\": R_combined,\n",
        "        },\n",
        "    )\n",
        "    \n",
        "    # Store results\n",
        "    time_hist.append(tk)\n",
        "    reference_hist.append(rk.flatten())\n",
        "    true_state_hist.append(xk.flatten())\n",
        "    estimate_hist.append(xhat.flatten())\n",
        "    measurement_mocap_hist.append(yk_combined[:3].flatten())\n",
        "    measurement_baro_hist.append(yk_combined[3].item())\n",
        "    command_hist.append(uk.flatten())\n",
        "    error_hist.append((xk - xhat).flatten())\n",
        "\n",
        "# Convert to numpy arrays\n",
        "reference_hist = np.array(reference_hist)\n",
        "true_state_hist = np.array(true_state_hist)\n",
        "estimate_hist = np.array(estimate_hist)\n",
        "measurement_mocap_hist = np.array(measurement_mocap_hist)\n",
        "measurement_baro_hist = np.array(measurement_baro_hist)\n",
        "command_hist = np.array(command_hist)\n",
        "error_hist = np.array(error_hist)\n"
    ]
})

# Visualize
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-sim-viz-header",
    "metadata": {},
    "source": [
        "### Visualize\n",
        "\n",
        "We can visualize the pertinent values of our system to assure correct behavior. Note how the Kalman filter fuses three heterogeneous sensors to provide accurate state estimates despite each sensor's limitations.\n"
    ]
})

new_cells.append({
    "cell_type": "code",
    "id": "cell-sim-viz",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "from mpl_toolkits.mplot3d import Axes3D\n",
        "\n",
        "fig = plt.figure(figsize=(15, 10))\n",
        "\n",
        "# Plot 1: 3D Trajectory\n",
        "ax = fig.add_subplot(2, 3, 1, projection='3d')\n",
        "ax.plot(\n",
        "    true_state_hist[:, 0],\n",
        "    true_state_hist[:, 1],\n",
        "    true_state_hist[:, 2],\n",
        "    'b-',\n",
        "    label='True',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.plot(\n",
        "    estimate_hist[:, 0],\n",
        "    estimate_hist[:, 1],\n",
        "    estimate_hist[:, 2],\n",
        "    'r--',\n",
        "    label='Estimate',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.scatter(\n",
        "    reference_hist[:, 0],\n",
        "    reference_hist[:, 1],\n",
        "    reference_hist[:, 2],\n",
        "    c='green',\n",
        "    s=100,\n",
        "    marker='*',\n",
        "    label='Setpoints'\n",
        ")\n",
        "ax.set_xlabel('X (m)')\n",
        "ax.set_ylabel('Y (m)')\n",
        "ax.set_zlabel('Z (m)')\n",
        "ax.set_title('3D Trajectory', fontweight='bold')\n",
        "ax.legend()\n",
        "\n",
        "# Plot 2: XY Trajectory (top view)\n",
        "ax = fig.add_subplot(2, 3, 2)\n",
        "ax.plot(\n",
        "    true_state_hist[:, 0],\n",
        "    true_state_hist[:, 1],\n",
        "    'b-',\n",
        "    label='True',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.plot(\n",
        "    estimate_hist[:, 0],\n",
        "    estimate_hist[:, 1],\n",
        "    'r--',\n",
        "    label='Estimate',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.scatter(\n",
        "    reference_hist[:, 0],\n",
        "    reference_hist[:, 1],\n",
        "    c='green',\n",
        "    s=100,\n",
        "    marker='*',\n",
        "    label='Setpoints',\n",
        "    zorder=5\n",
        ")\n",
        "ax.set_xlabel('X Position (m)', fontsize=11)\n",
        "ax.set_ylabel('Y Position (m)', fontsize=11)\n",
        "ax.set_title('XY Trajectory (Top View)', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "ax.axis('equal')\n",
        "\n",
        "# Plot 3: Z Position vs Time (with all sensors)\n",
        "ax = fig.add_subplot(2, 3, 3)\n",
        "ax.plot(time_hist, true_state_hist[:, 2], 'b-', label='True Z', linewidth=2, alpha=0.7)\n",
        "ax.plot(time_hist, estimate_hist[:, 2], 'r--', label='Estimated Z', linewidth=2, alpha=0.7)\n",
        "ax.scatter(\n",
        "    time_hist[::100],\n",
        "    measurement_mocap_hist[::100, 2],\n",
        "    c='green',\n",
        "    s=5,\n",
        "    alpha=0.3,\n",
        "    label='Mocap Z'\n",
        ")\n",
        "ax.scatter(\n",
        "    time_hist[::50],\n",
        "    measurement_baro_hist[::50],\n",
        "    c='purple',\n",
        "    s=5,\n",
        "    alpha=0.3,\n",
        "    label='Baro Z'\n",
        ")\n",
        "ax.set_ylabel('Z Position (m)', fontsize=11)\n",
        "ax.set_title('Altitude: Multi-Sensor Fusion', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Plot 4: X Position vs Time\n",
        "ax = fig.add_subplot(2, 3, 4)\n",
        "ax.plot(time_hist, true_state_hist[:, 0], 'b-', label='True X', alpha=0.7)\n",
        "ax.plot(time_hist, estimate_hist[:, 0], 'r--', label='Estimated X', alpha=0.7)\n",
        "ax.scatter(\n",
        "    time_hist[::100],\n",
        "    measurement_mocap_hist[::100, 0],\n",
        "    c='gray',\n",
        "    s=5,\n",
        "    alpha=0.3,\n",
        "    label='Mocap'\n",
        ")\n",
        "ax.set_ylabel('X Position (m)', fontsize=11)\n",
        "ax.set_xlabel('Time (s)', fontsize=11)\n",
        "ax.set_title('X Position Over Time', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Plot 5: Velocity Commands\n",
        "ax = fig.add_subplot(2, 3, 5)\n",
        "ax.plot(time_hist, command_hist[:, 0], label='Vx cmd', alpha=0.7)\n",
        "ax.plot(time_hist, command_hist[:, 1], label='Vy cmd', alpha=0.7)\n",
        "ax.plot(time_hist, command_hist[:, 2], label='Vz cmd', alpha=0.7)\n",
        "ax.set_xlabel('Time (s)', fontsize=11)\n",
        "ax.set_ylabel('Velocity Command (m/s)', fontsize=11)\n",
        "ax.set_title('Control Commands', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Plot 6: Position Estimation Error\n",
        "ax = fig.add_subplot(2, 3, 6)\n",
        "position_error = np.sqrt(\n",
        "    error_hist[:, 0]**2 + error_hist[:, 1]**2 + error_hist[:, 2]**2\n",
        ")\n",
        "ax.plot(time_hist, position_error * 1000, 'm-', linewidth=1.5)  # Convert to mm\n",
        "ax.set_xlabel('Time (s)', fontsize=11)\n",
        "ax.set_ylabel('Position Error (mm)', fontsize=11)\n",
        "ax.set_title('3D Position Estimation Error', fontsize=13, fontweight='bold')\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"Mean position error: {np.mean(position_error)*1000:.2f} mm\")\n",
        "print(f\"Max position error: {np.max(position_error)*1000:.2f} mm\")\n"
    ]
})

# Callback Wrapper section
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-callback-header",
    "metadata": {},
    "source": [
        "## Callback Wrapper\n",
        "\n",
        "Just as we did with the cruise control and TurtleBot systems, we can wrap the **entire Crazyflie navigation system** in a single callback. This encapsulates all five dynamical systems (setpoint generator, position controller, plant, multi-sensor array, and observer) and their states, making it easy to reuse and experiment with different configurations. This pattern is particularly useful for integration with ROS2, as we'll see in later tutorials.\n"
    ]
})

# Keep the callback implementation from original (cell 17)
if len(nb['cells']) > 17:
    # Update the callback cell with proper imports
    callback_cell = nb['cells'][17].copy()
    # Fix the callback to use proper imports
    callback_source = ''.join(callback_cell.get('source', []))
    # Replace KF.h with KF_module.h if needed
    callback_source = callback_source.replace('KF.h', 'KF_module.h')
    callback_cell['source'] = [callback_source]
    callback_cell['id'] = 'cell-callback-impl'
    new_cells.append(callback_cell)

# Add callback demonstration
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-callback-demo-header",
    "metadata": {},
    "source": [
        "Now we can run the same simulation with a much cleaner interface. All system parameters can be configured at initialization, and the simulation loop becomes trivial:\n"
    ]
})

new_cells.append({
    "cell_type": "code",
    "id": "cell-callback-demo",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Initialize the complete Crazyflie navigation system\n",
        "np.random.seed(42)\n",
        "crazyflie_system = initialize_crazyflie_system(\n",
        "    # Initial states\n",
        "    xk_init=np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]),\n",
        "    xhat_init=np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]),\n",
        "    P_init=np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0]),\n",
        "    # System parameters\n",
        "    dt=0.01,\n",
        "    Kp=0.8,\n",
        "    # Kalman filter parameters\n",
        "    Q=np.diag([0.001, 0.001, 0.001, 0.1, 0.1, 0.1]),\n",
        "    R_mocap=np.diag([0.005, 0.005, 0.005]),\n",
        "    R_baro=np.array([[0.1]]),\n",
        "    R_imu=np.diag([0.1, 0.1, 0.1]),\n",
        "    # Setpoint parameters\n",
        "    initial_hover=np.array([[0.0], [0.0], [1.0]]),\n",
        "    transitions=[\n",
        "        (20.0, [1.0, 1.0, 1.5]),\n",
        "        (40.0, [0.0, 0.0, 1.0]),\n",
        "    ],\n",
        ")\n",
        "\n",
        "# Simulation time\n",
        "T_sim = 60.0\n",
        "dt = 0.01\n",
        "time_steps = np.arange(0, T_sim, dt)\n",
        "\n",
        "# Storage for results\n",
        "results = {\n",
        "    \"time\": [],\n",
        "    \"reference\": [],\n",
        "    \"command\": [],\n",
        "    \"true_state\": [],\n",
        "    \"measurement_mocap\": [],\n",
        "    \"measurement_baro\": [],\n",
        "    \"measurement_imu\": [],\n",
        "    \"estimate\": [],\n",
        "    \"estimation_error\": [],\n",
        "}\n",
        "\n",
        "# Run simulation\n",
        "for tk in time_steps:\n",
        "    output = crazyflie_system(tk)\n",
        "\n",
        "    results[\"time\"].append(output[\"time\"])\n",
        "    results[\"reference\"].append(output[\"reference\"])\n",
        "    results[\"command\"].append(output[\"command\"])\n",
        "    results[\"true_state\"].append(output[\"true_state\"])\n",
        "    results[\"measurement_mocap\"].append(output[\"measurement_mocap\"])\n",
        "    results[\"measurement_baro\"].append(output[\"measurement_baro\"])\n",
        "    results[\"measurement_imu\"].append(output[\"measurement_imu\"])\n",
        "    results[\"estimate\"].append(output[\"estimate\"])\n",
        "    results[\"estimation_error\"].append(output[\"estimation_error\"])\n",
        "\n",
        "# Convert to numpy arrays for plotting\n",
        "for key in [\"reference\", \"command\", \"true_state\", \"measurement_mocap\", \n",
        "            \"measurement_imu\", \"estimate\", \"estimation_error\"]:\n",
        "    results[key] = np.array(results[key])\n",
        "results[\"measurement_baro\"] = np.array(results[\"measurement_baro\"])\n"
    ]
})

# Add callback visualization (similar to direct simulation plots)
new_cells.append({
    "cell_type": "code",
    "id": "cell-callback-viz",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Visualize results from callback\n",
        "fig = plt.figure(figsize=(15, 10))\n",
        "\n",
        "# Plot 1: 3D Trajectory\n",
        "ax = fig.add_subplot(2, 3, 1, projection='3d')\n",
        "ax.plot(\n",
        "    results[\"true_state\"][:, 0],\n",
        "    results[\"true_state\"][:, 1],\n",
        "    results[\"true_state\"][:, 2],\n",
        "    'b-',\n",
        "    label='True',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.plot(\n",
        "    results[\"estimate\"][:, 0],\n",
        "    results[\"estimate\"][:, 1],\n",
        "    results[\"estimate\"][:, 2],\n",
        "    'r--',\n",
        "    label='Estimate',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.scatter(\n",
        "    results[\"reference\"][:, 0],\n",
        "    results[\"reference\"][:, 1],\n",
        "    results[\"reference\"][:, 2],\n",
        "    c='green',\n",
        "    s=100,\n",
        "    marker='*',\n",
        "    label='Setpoints'\n",
        ")\n",
        "ax.set_xlabel('X (m)')\n",
        "ax.set_ylabel('Y (m)')\n",
        "ax.set_zlabel('Z (m)')\n",
        "ax.set_title('3D Trajectory (Callback Interface)', fontweight='bold')\n",
        "ax.legend()\n",
        "\n",
        "# Plot 2: XY Trajectory (top view)\n",
        "ax = fig.add_subplot(2, 3, 2)\n",
        "ax.plot(\n",
        "    results[\"true_state\"][:, 0],\n",
        "    results[\"true_state\"][:, 1],\n",
        "    'b-',\n",
        "    label='True',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.plot(\n",
        "    results[\"estimate\"][:, 0],\n",
        "    results[\"estimate\"][:, 1],\n",
        "    'r--',\n",
        "    label='Estimate',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.scatter(\n",
        "    results[\"reference\"][:, 0],\n",
        "    results[\"reference\"][:, 1],\n",
        "    c='green',\n",
        "    s=100,\n",
        "    marker='*',\n",
        "    label='Setpoints',\n",
        "    zorder=5\n",
        ")\n",
        "ax.set_xlabel('X Position (m)', fontsize=11)\n",
        "ax.set_ylabel('Y Position (m)', fontsize=11)\n",
        "ax.set_title('XY Trajectory (Top View)', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "ax.axis('equal')\n",
        "\n",
        "# Plot 3: Z Position vs Time (with all sensors)\n",
        "ax = fig.add_subplot(2, 3, 3)\n",
        "ax.plot(\n",
        "    results[\"time\"],\n",
        "    results[\"true_state\"][:, 2],\n",
        "    'b-',\n",
        "    label='True Z',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.plot(\n",
        "    results[\"time\"],\n",
        "    results[\"estimate\"][:, 2],\n",
        "    'r--',\n",
        "    label='Estimated Z',\n",
        "    linewidth=2,\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.scatter(\n",
        "    results[\"time\"][::100],\n",
        "    results[\"measurement_mocap\"][::100, 2],\n",
        "    c='green',\n",
        "    s=5,\n",
        "    alpha=0.3,\n",
        "    label='Mocap Z'\n",
        ")\n",
        "ax.scatter(\n",
        "    results[\"time\"][::50],\n",
        "    results[\"measurement_baro\"][::50],\n",
        "    c='purple',\n",
        "    s=5,\n",
        "    alpha=0.3,\n",
        "    label='Baro Z'\n",
        ")\n",
        "ax.set_ylabel('Z Position (m)', fontsize=11)\n",
        "ax.set_title('Altitude: Multi-Sensor Fusion', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Plot 4: X Position vs Time\n",
        "ax = fig.add_subplot(2, 3, 4)\n",
        "ax.plot(\n",
        "    results[\"time\"],\n",
        "    results[\"true_state\"][:, 0],\n",
        "    'b-',\n",
        "    label='True X',\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.plot(\n",
        "    results[\"time\"],\n",
        "    results[\"estimate\"][:, 0],\n",
        "    'r--',\n",
        "    label='Estimated X',\n",
        "    alpha=0.7\n",
        ")\n",
        "ax.scatter(\n",
        "    results[\"time\"][::100],\n",
        "    results[\"measurement_mocap\"][::100, 0],\n",
        "    c='gray',\n",
        "    s=5,\n",
        "    alpha=0.3,\n",
        "    label='Mocap'\n",
        ")\n",
        "ax.set_ylabel('X Position (m)', fontsize=11)\n",
        "ax.set_xlabel('Time (s)', fontsize=11)\n",
        "ax.set_title('X Position Over Time', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Plot 5: Velocity Commands\n",
        "ax = fig.add_subplot(2, 3, 5)\n",
        "ax.plot(results[\"time\"], results[\"command\"][:, 0], label='Vx cmd', alpha=0.7)\n",
        "ax.plot(results[\"time\"], results[\"command\"][:, 1], label='Vy cmd', alpha=0.7)\n",
        "ax.plot(results[\"time\"], results[\"command\"][:, 2], label='Vz cmd', alpha=0.7)\n",
        "ax.set_xlabel('Time (s)', fontsize=11)\n",
        "ax.set_ylabel('Velocity Command (m/s)', fontsize=11)\n",
        "ax.set_title('Control Commands', fontsize=13, fontweight='bold')\n",
        "ax.legend()\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Plot 6: Position Estimation Error\n",
        "ax = fig.add_subplot(2, 3, 6)\n",
        "position_error = np.sqrt(\n",
        "    results[\"estimation_error\"][:, 0]**2 + \n",
        "    results[\"estimation_error\"][:, 1]**2 + \n",
        "    results[\"estimation_error\"][:, 2]**2\n",
        ")\n",
        "ax.plot(\n",
        "    results[\"time\"],\n",
        "    position_error * 1000,\n",
        "    'm-',\n",
        "    linewidth=1.5\n",
        ")  # Convert to mm\n",
        "ax.set_xlabel('Time (s)', fontsize=11)\n",
        "ax.set_ylabel('Position Error (mm)', fontsize=11)\n",
        "ax.set_title('3D Position Estimation Error', fontsize=13, fontweight='bold')\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"Mean position error: {np.mean(position_error)*1000:.2f} mm\")\n",
        "print(f\"Max position error: {np.max(position_error)*1000:.2f} mm\")\n"
    ]
})

# Experimentation
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-experiment",
    "metadata": {},
    "source": [
        "## Experimentation\n",
        "\n",
        "Now that we have a working system, try experimenting with different parameters:\n",
        "\n",
        "**Sensor characteristics**:\n",
        "- Vary mocap noise to simulate poor lighting conditions\n",
        "- Increase barometer noise to simulate temperature-induced drift\n",
        "- Increase IMU noise to simulate cheap sensors\n",
        "- Observe how the KF automatically reweights sensors\n",
        "\n",
        "**Controller tuning**:\n",
        "- Adjust `Kp` to change position tracking aggressiveness\n",
        "- Try different hover patterns (figure-eight, circle, random)\n",
        "- Add velocity limits to simulate real quadrotor constraints\n",
        "\n",
        "**Observer comparison**:\n",
        "- Run simulation with KF disabled (use raw mocap only)\n",
        "- Compare estimation error with single-sensor vs multi-sensor\n",
        "- Visualize which sensor dominates in different flight regimes\n",
        "\n",
        "**Challenge**: Design a trajectory where fusion of all three sensors significantly outperforms any single sensor.\n"
    ]
})

# Final cell: Parent link
new_cells.append({
    "cell_type": "markdown",
    "id": "cell-end",
    "metadata": {},
    "source": [
        "[← Control Systems as Dynamical Systems](../../../getting_started/theory_to_python/control_systems_as_dynamical_systems.rst)\n"
    ]
})

# Write the restructured notebook
nb['cells'] = new_cells

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully restructured {nb_path}")
print(f"Total cells: {len(new_cells)}")
