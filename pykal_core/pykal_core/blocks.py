from __future__ import annotations

# Stdlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

# Third-party
import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray

# Local
from pykal_core.utils.control_system.safeio import SafeIO


class DSBlock:

    def __init__(
        self,
        *,
        f: Optional[Callable] = None,
        h: Optional[Callable] = None,
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
        sys_type: Optional[str] = None,
    ) -> None:

        # initialize state space and measurement space

        self._X = []
        self._Y = []

        # set dynamics function f and output map h
        def zero_dynamics(xk: NDArray) -> NDArray:
            return np.zeros_like(xk)

        self._f = SafeIO._validate_func_signature(f) if f is not None else zero_dynamics

        def identity_map(xk: NDArray) -> NDArray:
            return xk

        self._h = SafeIO._validate_func_signature(h) if h is not None else identity_map

        # set process noise and output noise matrix functions

        self._Q = SafeIO._validate_func_signature(Q) if Q is not None else None
        self._R = SafeIO._validate_func_signature(R) if R is not None else None

        self._sys_type = (
            sys_type if sys_type is not None else ("cti" if f is not None else None)
        )

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @X.setter
    def X(self,x):
        self._X = x

    @Y.setter
    def Y(self,y):
        self._Y = y

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        self._f = SafeIO._validate_func_signature(f)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self._h = SafeIO._validate_func_signature(h)

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = SafeIO._validate_func_signature(Q)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = SafeIO._validate_func_signature(R)

    @property
    def sys_type(self):
        return self._sys_type

    @sys_type.setter
    def sys_type(self, sys_type):
        self._sys_type = sys_type





# --- Default converters (Float64MultiArray <-> np.ndarray) --------------------
def _default_ros2py(msg: Float64MultiArray) -> np.ndarray:
    return np.asarray(msg.data, dtype=float)


def _default_py2ros(arr: np.ndarray) -> Float64MultiArray:
    return Float64MultiArray(data=np.asarray(arr, dtype=float).ravel().tolist())


class ROSBlock(Node):
    r"""
    Unified ROS2 block wrapping a NumPy-based system function.

    This class merges the responsibilities of a **ROS2 node** and a
    **runner/manager** into a single object. It subscribes to one or more
    topics, converts incoming ROS messages into NumPy arrays, feeds them to a
    user-defined system function, and publishes the resulting NumPy arrays
    back as ROS messages.

    ## Purpose
    - Bridge ROS2 topics with a Python/NumPy system function:

      .. code-block:: python

         def system_function(tk: float, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
             ...

      where `tk` is elapsed ROS time (s), and `inputs` maps subscribed
      topic names to NumPy arrays.

    ## Constructor
    ```
    ROSBlock(
        node_name: str,
        system_wrapper: Callable[[float, Dict[str, np.ndarray]], Dict[str, np.ndarray]],
        subscribes_to: Optional[List[str]] = None,
        publishes_to: Optional[List[str]] = None,
        sub_msg_types: Optional[Dict[str, type]] = None,
        pub_msg_types: Optional[Dict[str, type]] = None,
        ros2py_dict: Optional[Dict[str, Callable[[Any], np.ndarray]]] = None,
        py2ros_dict: Optional[Dict[str, Callable[[np.ndarray], Any]]] = None,
        rate_hz: float = 10.0,
        use_single_threaded_executor: bool = True,
        shutdown_on_stop: bool = True,
        qos_profile: Optional[QoSProfile] = None,
    )
    ```

    ## Lifecycle Methods
    - `start()`: spin in a background thread and periodically invoke `system_wrapper`.
    - `stop()`: stop executor, destroy node, optional `rclpy.shutdown()`.
    - `latest_inputs() -> Dict[str, np.ndarray]`: most recent inputs by topic.
    - `is_running() -> bool`: whether the executor thread is alive.
    - `node()`: returns `self` (for direct Node APIs).

    ## Behavior
    - Subscribed messages are converted via `ros2py_dict[topic]` (default provided)
      into NumPy arrays and cached.
    - Every tick (1 / `rate_hz`), a snapshot of inputs and current ROS time `tk`
      are passed to `system_wrapper`. It must return a dict of topic -> np.ndarray.
    - Arrays are converted to ROS messages with `py2ros_dict[topic]` and published.

    ## Notes
    - Defaults to `Float64MultiArray` for msg types and simple array flattening.
    - Per-topic dicts are checked against the declared topic lists.
    - Unknown topics in `system_wrapper` output are ignored.
    """

    def __init__(
        self,
        node_name: str,
        system_wrapper: Callable[[float, Dict[str, np.ndarray]], Dict[str, np.ndarray]],
        *,
        subscribes_to: Optional[Sequence[str]] = None,
        publishes_to: Optional[Sequence[str]] = None,
        sub_msg_types: Optional[Dict[str, type]] = None,
        pub_msg_types: Optional[Dict[str, type]] = None,
        ros2py_dict: Optional[Dict[str, Callable[[Any], np.ndarray]]] = None,
        py2ros_dict: Optional[Dict[str, Callable[[np.ndarray], Any]]] = None,
        rate_hz: float = 10.0,
        use_single_threaded_executor: bool = True,
        shutdown_on_stop: bool = True,
        qos_profile: Optional[QoSProfile] = None,
    ) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)        
        # --- Node init
        super().__init__(node_name)

        # --- Store core
        self._system_wrapper = system_wrapper
        self._subs: List[str] = list(subscribes_to or [])
        self._pubs: List[str] = list(publishes_to or [])
        self._rate_hz = float(rate_hz)
        if not (self._rate_hz > 0):
            raise ValueError("rate_hz must be positive")

        # --- Executor/thread config
        self._use_single = bool(use_single_threaded_executor)
        self._shutdown_on_stop = bool(shutdown_on_stop)
        self._executor = None
        self._thread: Optional[threading.Thread] = None

        # --- Validation: duplicates
        if len(set(self._subs)) != len(self._subs):
            dup = sorted({t for t in self._subs if self._subs.count(t) > 1})
            raise ValueError(f"Duplicate subscribe topics: {dup}")
        if len(set(self._pubs)) != len(self._pubs):
            dup = sorted({t for t in self._pubs if self._pubs.count(t) > 1})
            raise ValueError(f"Duplicate publish topics: {dup}")

        # --- QoS / callback group
        self._qos = qos_profile or QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=10, reliability=QoSReliabilityPolicy.RELIABLE
        )
        self._cb_group = ReentrantCallbackGroup()

        # --- Per-topic message types (default Float64MultiArray)
        sub_msg_types = dict(sub_msg_types or {})
        pub_msg_types = dict(pub_msg_types or {})
        extra_sub = set(sub_msg_types) - set(self._subs)
        extra_pub = set(pub_msg_types) - set(self._pubs)
        if extra_sub:
            raise ValueError(f"sub_msg_types has unknown topics: {sorted(extra_sub)}")
        if extra_pub:
            raise ValueError(f"pub_msg_types has unknown topics: {sorted(extra_pub)}")

        self._sub_msg_types: Dict[str, type] = {t: sub_msg_types.get(t, Float64MultiArray) for t in self._subs}
        self._pub_msg_types: Dict[str, type] = {t: pub_msg_types.get(t, Float64MultiArray) for t in self._pubs}
        for t, typ in {**self._sub_msg_types, **self._pub_msg_types}.items():
            if not isinstance(typ, type):
                raise TypeError(f"Message type for topic '{t}' must be a class; got {type(typ).__name__}")

        # --- Converters (defaults provided)
        ros2py_dict = dict(ros2py_dict or {})
        py2ros_dict = dict(py2ros_dict or {})
        extra_ros2py = set(ros2py_dict) - set(self._subs)
        extra_py2ros = set(py2ros_dict) - set(self._pubs)
        if extra_ros2py:
            raise ValueError(f"ros2py_dict has unknown topics: {sorted(extra_ros2py)}")
        if extra_py2ros:
            raise ValueError(f"py2ros_dict has unknown topics: {sorted(extra_py2ros)}")

        self._ros2py: Dict[str, Callable[[Any], np.ndarray]] = {
            t: ros2py_dict.get(t, _default_ros2py) for t in self._subs
        }
        self._py2ros: Dict[str, Callable[[np.ndarray], Any]] = {
            t: py2ros_dict.get(t, _default_py2ros) for t in self._pubs
        }
        for t, fn in {**self._ros2py, **self._py2ros}.items():
            if not callable(fn):
                raise TypeError(f"Converter for topic '{t}' must be callable")

        # --- Internal state
        self._inputs_lock = threading.Lock()
        self._inputs: Dict[str, np.ndarray] = {}

        # --- Publishers
        self._pub_map: Dict[str, Any] = {}
        for topic in self._pubs:
            msg_type = self._pub_msg_types[topic]
            pub = self.create_publisher(msg_type, topic, self._qos)
            self._pub_map[topic] = pub
            # smoke test py2ros
            conv = self._py2ros[topic]
            try:
                test_msg = conv(np.zeros(1))
                if not isinstance(test_msg, msg_type):
                    self.get_logger().warning(
                        f"[{topic}] py2ros produced {type(test_msg).__name__}, expected {msg_type.__name__}"
                    )
            except Exception as e:
                self.get_logger().warning(f"[{topic}] py2ros smoke test failed: {e}")

        # --- Subscriptions
        for topic in self._subs:
            msg_type = self._sub_msg_types[topic]
            self.create_subscription(
                msg_type, topic, self._make_sub_cb(topic), self._qos, callback_group=self._cb_group
            )
            if not callable(self._ros2py[topic]):
                self.get_logger().warning(f"[{topic}] ros2py converter is not callable")

        # --- Timer (ROS clock)
        self._t0 = self.get_clock().now()
        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer, callback_group=self._cb_group)

        self.get_logger().info(
            f"[{self.get_name()}] subs={self._subs}, pubs={self._pubs}, rate={self._rate_hz} Hz"
        )

    # ------------------------------ Callbacks ---------------------------------
    def _make_sub_cb(self, topic: str):
        def _cb(msg):
            try:
                conv = self._ros2py[topic]
                arr = np.asarray(conv(msg), dtype=float)
                if arr.ndim == 0:
                    arr = arr[None]
                with self._inputs_lock:
                    self._inputs[topic] = arr
            except Exception as e:
                self.get_logger().error(f"[{topic}] ros2py conversion failed: {e}")
        return _cb

    def _on_timer(self):
        # elapsed ROS time (seconds)
        tk = (self.get_clock().now() - self._t0).nanoseconds * 1e-9

        # snapshot inputs
        with self._inputs_lock:
            inputs_snapshot = {k: v.copy() for k, v in self._inputs.items()}

        # run system
        try:
            outputs = self._system_wrapper(tk, inputs_snapshot)
            if not outputs:
                return
            if not isinstance(outputs, dict):
                self.get_logger().warning("system_wrapper did not return a dict; ignoring.")
                return
        except Exception as e:
            self.get_logger().error(f"system_wrapper error: {e}")
            return
        unknown = set(outputs) - set(self._pub_map)
        if unknown:
            self.get_logger().warning(f"Ignoring outputs for unknown topics: {sorted(unknown)}")
            
        # publish known topics only
        for topic, pub in self._pub_map.items():
            if topic not in outputs:
                continue
            try:
                arr = np.asarray(outputs[topic], dtype=float).ravel()
                conv = self._py2ros[topic]
                msg = conv(arr)
                pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"[{topic}] py2ros publish failed: {e}")

    # ---------------------------- Public API ----------------------------------
    def start(self):
        """Start the ROS2 executor in a background thread and spin this node."""
        if self.is_running():
            return
        if not rclpy.ok():
            rclpy.init()
        executor = SingleThreadedExecutor() if self._use_single else rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self)
        self._executor = executor
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        print(
            f"[ROSBlock] started node='{self.get_name()}' pub={self._pubs} sub={self._subs} @ {self._rate_hz} Hz"
        )

    def stop(self):
        """Stop the executor, destroy the node, and optionally call rclpy.shutdown()."""
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Destroy self (Node resources) *after* executor stops
        try:
            self.destroy_node()
        except Exception:
            pass

        if self._shutdown_on_stop:
            try:
                rclpy.shutdown()
            except Exception:
                pass

        print("[ROSBlock] stopped.")

    def latest_inputs(self) -> Dict[str, np.ndarray]:
        """Return most recent input arrays per topic (copy)."""
        with self._inputs_lock:
            return {k: np.asarray(v) for k, v in self._inputs.items()}

    def is_running(self) -> bool:
        """True if executor thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def node(self) -> "ROSBlock":
        """Return the underlying Node (self)."""
        return self


