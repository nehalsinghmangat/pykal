"""
Gazebo launcher utilities for pykal.

This module provides functions to launch and manage Gazebo simulations
directly from Jupyter notebooks, enabling self-contained robot simulations.
"""

import subprocess
import time
import signal
import os
from typing import Optional, Dict, Any, List
import warnings


class GazeboProcess:
    """
    Wrapper for a Gazebo subprocess with lifecycle management.

    Attributes:
        process: The subprocess.Popen object
        robot: Name of the robot being simulated
        world: Name of the world being used
        headless: Whether running in headless mode
    """

    def __init__(self, process: subprocess.Popen, robot: str, world: str, headless: bool):
        self.process = process
        self.robot = robot
        self.world = world
        self.headless = headless
        self._is_running = True

    def is_running(self) -> bool:
        """Check if Gazebo process is still running."""
        if not self._is_running:
            return False
        if self.process.poll() is not None:
            self._is_running = False
        return self._is_running

    def terminate(self) -> None:
        """Gracefully terminate the Gazebo process and all children."""
        if not self.is_running():
            return

        # Get the process group ID
        try:
            pgid = os.getpgid(self.process.pid)
        except (ProcessLookupError, AttributeError):
            pgid = None

        # Try graceful termination of the entire process group
        if pgid is not None and os.name != 'nt':
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Process group already gone
        else:
            # Fallback for Windows or if pgid not available
            self.process.terminate()

        # Wait for processes to terminate
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill the process group
            if pgid is not None and os.name != 'nt':
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self.process.kill()

            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

        # Final cleanup: kill any remaining gazebo/gz processes
        # This catches any orphaned processes that escaped the process group
        try:
            subprocess.run(['pkill', '-9', '-f', 'gz sim'],
                          stderr=subprocess.DEVNULL, timeout=2)
            subprocess.run(['pkill', '-9', '-f', 'gzserver'],
                          stderr=subprocess.DEVNULL, timeout=2)
            subprocess.run(['pkill', '-9', '-f', 'gzclient'],
                          stderr=subprocess.DEVNULL, timeout=2)
            subprocess.run(['pkill', '-9', '-f', 'ros_gz_bridge'],
                          stderr=subprocess.DEVNULL, timeout=2)
            subprocess.run(['pkill', '-9', '-f', 'ros_gz_sim'],
                          stderr=subprocess.DEVNULL, timeout=2)
            subprocess.run(['pkill', '-9', '-f', 'robot_state_publisher'],
                          stderr=subprocess.DEVNULL, timeout=2)
            time.sleep(0.5)  # Give processes time to die
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # pkill not available or timed out

        self._is_running = False

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        mode = "headless" if self.headless else "GUI"
        return f"GazeboProcess(robot={self.robot}, world={self.world}, mode={mode}, status={status})"


def start_gazebo(
    robot: str = 'turtlebot3',
    world: Optional[str] = None,
    headless: bool = False,
    x_pose: float = 0.0,
    y_pose: float = 0.0,
    z_pose: float = 0.0,
    yaw: float = 0.0,
    launch_file: Optional[str] = None,
    package: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    model: str = 'burger'
) -> GazeboProcess:
    """
    Launch Gazebo simulation with a robot.

    This function starts a Gazebo simulation that can be used alongside
    pykal ROSNode instances in Jupyter notebooks for complete
    self-contained robot simulations.

    Parameters
    ----------
    robot : str, default='turtlebot3'
        Robot type to simulate. Options:
        - 'turtlebot3': TurtleBot3 (Burger or Waffle)
        - 'crazyflie': Crazyflie quadrotor
        - 'custom': Use custom launch_file and package
    world : str, optional
        World/environment name. If None, uses default for robot type:
        - turtlebot3: 'turtlebot3_world'
        - crazyflie: 'crazyflie_world'
    headless : bool, default=False
        If True, run Gazebo without GUI (faster, less resource intensive)
    x_pose : float, default=0.0
        Initial x position of robot (meters)
    y_pose : float, default=0.0
        Initial y position of robot (meters)
    z_pose : float, default=0.0
        Initial z position of robot (meters)
    yaw : float, default=0.0
        Initial yaw angle of robot (radians)
    launch_file : str, optional
        Custom launch file name (required if robot='custom')
    package : str, optional
        ROS2 package containing launch file (required if robot='custom')
    extra_args : dict, optional
        Additional launch arguments as key-value pairs
    verbose : bool, default=True
        Print status messages
    model : str, default='burger'
        TurtleBot3 model type (burger, waffle, or waffle_pi).
        Only used when robot='turtlebot3'

    Returns
    -------
    GazeboProcess
        Process wrapper that can be used to manage the Gazebo instance

    Examples
    --------
    Launch TurtleBot3 with GUI::

        >>> from pykal.utilities.gazebo import start_gazebo, stop_gazebo
        >>> gz = start_gazebo(robot='turtlebot3', world='turtlebot3_world')
        >>> # ... run your simulation ...
        >>> stop_gazebo(gz)

    Launch Crazyflie in headless mode::

        >>> gz = start_gazebo(
        ...     robot='crazyflie',
        ...     headless=True,
        ...     z_pose=0.5
        ... )

    Launch custom robot::

        >>> gz = start_gazebo(
        ...     robot='custom',
        ...     package='my_robot_gazebo',
        ...     launch_file='my_robot.launch.py',
        ...     extra_args={'use_sim_time': 'true'}
        ... )

    Raises
    ------
    ValueError
        If invalid robot type or missing required arguments
    RuntimeError
        If Gazebo fails to start
    """
    # Validate inputs
    valid_robots = ['turtlebot3', 'crazyflie', 'custom']
    if robot not in valid_robots:
        raise ValueError(f"robot must be one of {valid_robots}, got '{robot}'")

    if robot == 'custom' and (launch_file is None or package is None):
        raise ValueError("launch_file and package are required when robot='custom'")

    # Set defaults based on robot type
    if world is None:
        world_defaults = {
            'turtlebot3': 'turtlebot3_world',
            'crazyflie': 'crazyflie_world',
            'custom': 'empty_world'
        }
        world = world_defaults[robot]

    # Build launch command
    cmd = ['ros2', 'launch']

    if robot == 'turtlebot3':
        cmd.extend(['turtlebot3_gazebo', f'{world}.launch.py'])
    elif robot == 'crazyflie':
        cmd.extend(['crazyflie_gazebo', f'{world}.launch.py'])
    else:  # custom
        cmd.extend([package, launch_file])

    # Add pose arguments
    cmd.extend([
        f'x_pose:={x_pose}',
        f'y_pose:={y_pose}',
        f'z_pose:={z_pose}',
        f'yaw:={yaw}'
    ])

    # Add GUI flag (explicit for both headless and GUI mode)
    if headless:
        cmd.append('gui:=false')
    else:
        cmd.append('gui:=true')  # Explicitly enable GUI

    # Add extra arguments
    if extra_args:
        for key, value in extra_args.items():
            cmd.append(f'{key}:={value}')

    if verbose:
        print(f"Launching Gazebo: {robot} in {world}")
        print(f"Mode: {'headless' if headless else 'GUI'}")
        print(f"Command: {' '.join(cmd)}")

    # Set up environment variables
    env = os.environ.copy()

    # GPU/Rendering fixes for black window issue
    if not headless:
        # Try software rendering first (fixes most black window issues)
        # Set to '1' for software rendering, '0' for hardware rendering
        # If you have a working GPU, try setting LIBGL_ALWAYS_SOFTWARE=0
        if 'LIBGL_ALWAYS_SOFTWARE' not in env:
            env['LIBGL_ALWAYS_SOFTWARE'] = '0'  # Try hardware first

        # Ensure DISPLAY is set
        if 'DISPLAY' not in env:
            env['DISPLAY'] = ':0'

        if verbose:
            print(f"Display: {env.get('DISPLAY', 'not set')}")
            print(f"GPU Mode: {'Software' if env.get('LIBGL_ALWAYS_SOFTWARE') == '1' else 'Hardware'}")

    if robot == 'turtlebot3':
        env['TURTLEBOT3_MODEL'] = model
        if verbose:
            print(f"TurtleBot3 model: {model}")

    # Ensure ROS2 is sourced if not already
    if 'ROS_DISTRO' not in env:
        # Try to find and source ROS2 setup
        ros_setup_paths = [
            '/opt/ros/jazzy/setup.bash',
            '/opt/ros/humble/setup.bash',
            '/opt/ros/iron/setup.bash'
        ]
        for setup_path in ros_setup_paths:
            if os.path.exists(setup_path):
                # We need to source the setup file - convert command to shell script
                shell_cmd = f'source {setup_path} && {" ".join(cmd)}'
                cmd = ['bash', '-c', shell_cmd]
                break

    # Launch Gazebo
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            preexec_fn=os.setsid if os.name != 'nt' else None,
            env=env
        )

        # Give Gazebo time to start and render
        if verbose:
            print("Waiting for Gazebo to initialize...")
            if not headless:
                print("  (Scene rendering may take 10-15 seconds...)")

        # Wait longer for GUI mode to allow rendering
        time.sleep(10 if not headless else 3)

        # Check if process started successfully
        if process.poll() is not None:
            raise RuntimeError(
                f"Gazebo failed to start. Return code: {process.returncode}"
            )

        if verbose:
            print("✓ Gazebo launched successfully")
            if not headless:
                print("\n" + "="*60)
                print("GAZEBO WINDOW TROUBLESHOOTING")
                print("="*60)
                print("If you see a BLACK WINDOW:")
                print("  1. Wait 10-20 seconds for scene to load")
                print("  2. Try moving/rotating view (middle-click drag)")
                print("  3. If still black, try software rendering:")
                print("     export LIBGL_ALWAYS_SOFTWARE=1")
                print("     (run this before starting Python/Jupyter)")
                print("  4. Check GPU drivers are installed")
                print("  5. Verify: glxinfo | grep 'OpenGL renderer'")
                print("="*60 + "\n")

        return GazeboProcess(process, robot, world, headless)

    except FileNotFoundError:
        raise RuntimeError(
            "ros2 command not found. Make sure ROS2 is installed and sourced."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to launch Gazebo: {e}")


def stop_gazebo(gz_process: GazeboProcess, verbose: bool = True) -> None:
    """
    Stop a running Gazebo simulation.

    Parameters
    ----------
    gz_process : GazeboProcess
        The Gazebo process to stop (returned by start_gazebo)
    verbose : bool, default=True
        Print status messages

    Examples
    --------
    >>> gz = start_gazebo(robot='turtlebot3')
    >>> # ... simulation code ...
    >>> stop_gazebo(gz)
    """
    if not isinstance(gz_process, GazeboProcess):
        raise TypeError("gz_process must be a GazeboProcess instance")

    if not gz_process.is_running():
        if verbose:
            print("Gazebo is already stopped")
        return

    if verbose:
        print(f"Stopping Gazebo ({gz_process.robot} in {gz_process.world})...")

    gz_process.terminate()

    if verbose:
        print("✓ Gazebo stopped")


def restart_gazebo(
    gz_process: GazeboProcess,
    **kwargs
) -> GazeboProcess:
    """
    Restart a Gazebo simulation with the same or updated parameters.

    Parameters
    ----------
    gz_process : GazeboProcess
        The Gazebo process to restart
    **kwargs
        Any parameters to override (same as start_gazebo)

    Returns
    -------
    GazeboProcess
        New Gazebo process instance

    Examples
    --------
    >>> gz = start_gazebo(robot='turtlebot3')
    >>> # ... simulation ...
    >>> gz = restart_gazebo(gz, x_pose=1.0, y_pose=1.0)  # Restart at new position
    """
    # Stop current instance
    stop_gazebo(gz_process, verbose=kwargs.get('verbose', True))

    # Preserve original settings unless overridden
    params = {
        'robot': gz_process.robot,
        'world': gz_process.world,
        'headless': gz_process.headless
    }
    params.update(kwargs)

    # Start new instance
    return start_gazebo(**params)


# Convenience function for common use case
def quick_start(robot: str = 'turtlebot3', headless: bool = False) -> GazeboProcess:
    """
    Quickly start Gazebo with default settings.

    Convenience function for common use case.

    Parameters
    ----------
    robot : str, default='turtlebot3'
        Robot type ('turtlebot3' or 'crazyflie')
    headless : bool, default=False
        Run without GUI

    Returns
    -------
    GazeboProcess
        Gazebo process instance

    Examples
    --------
    >>> from pykal.utilities.gazebo import quick_start, stop_gazebo
    >>> gz = quick_start('turtlebot3')
    >>> # ... your code ...
    >>> stop_gazebo(gz)
    """
    return start_gazebo(robot=robot, headless=headless)
