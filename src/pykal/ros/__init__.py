"""
ROS2 integration utilities for pykal.

This module provides message conversion utilities, ROSNode wrapper,
and helper functions for seamless integration with ROS2.
"""

# Lazy imports for ROS2 modules - only fail if user actually tries to use them
_ros2py_py2ros = None
_ros2py_import_attempted = False
_ros_node = None
_ros_node_import_attempted = False


def __getattr__(name):
    global _ros2py_py2ros, _ros2py_import_attempted
    global _ros_node, _ros_node_import_attempted

    if name == "ros2py_py2ros":
        if _ros2py_import_attempted:
            if _ros2py_py2ros is not None:
                return _ros2py_py2ros
            else:
                raise ImportError(
                    "ros2py_py2ros requires ROS2 dependencies. Install with: pip install pykal[ros2]"
                )

        _ros2py_import_attempted = True
        try:
            from . import ros2py_py2ros as _module

            _ros2py_py2ros = _module
            return _ros2py_py2ros
        except ImportError as e:
            raise ImportError(
                f"ros2py_py2ros requires ROS2 dependencies. Install with: pip install pykal[ros2]\n"
                f"Original error: {e}"
            ) from e

    if name == "ros_node":
        if _ros_node_import_attempted:
            if _ros_node is not None:
                return _ros_node
            else:
                raise ImportError(
                    "ros_node requires ROS2 dependencies. Install with: pip install pykal[ros2]"
                )

        _ros_node_import_attempted = True
        try:
            from . import ros_node as _module

            _ros_node = _module
            return _ros_node
        except ImportError as e:
            raise ImportError(
                f"ros_node requires ROS2 dependencies. Install with: pip install pykal[ros2]\n"
                f"Original error: {e}"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ros2py_py2ros", "ros_node"]
