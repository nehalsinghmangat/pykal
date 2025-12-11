"""
pykal: A modular Python framework for dynamical systems, estimation, and ROS2 integration.
"""

from .dynamical_system import DynamicalSystem

# Lazy import for ROSNode - only fails if user actually tries to use it
def __getattr__(name):
    if name == "ROSNode":
        try:
            from .ros.ros_node import ROSNode
            return ROSNode
        except ImportError as e:
            raise ImportError(
                f"ROSNode requires ROS2 dependencies. Install with: pip install pykal[ros2]\n"
                f"Original error: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Expose key modules as namespaces, not direct symbols
from . import state_estimators
from . import controllers
from . import ros
from . import gazebo
from . import data_change

__all__ = [
    "DynamicalSystem",
    "ROSNode",
    "state_estimators",
    "controllers",
    "ros",
    "gazebo",
    "data_change",
]
