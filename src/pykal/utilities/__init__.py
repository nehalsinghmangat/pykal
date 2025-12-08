"""
Utility modules for pykal: converters, estimator wrappers, and demonstrations.
"""

from . import estimators
from . import gazebo

# Lazy import for ros2py_py2ros - only fails if user actually tries to use it
_ros2py_py2ros = None
_ros2py_import_attempted = False

def __getattr__(name):
    global _ros2py_py2ros, _ros2py_import_attempted

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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "estimators",
    "gazebo",
    "ros2py_py2ros",
]
