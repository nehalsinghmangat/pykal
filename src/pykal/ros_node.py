from typing import Callable, Dict, Optional, Sequence, Tuple, Any, List
import numpy as np
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from numpy.typing import NDArray

from pykal.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT

SubTuple = Tuple[str, type, str]  # (topic, msg_type, arg_name)
PubTuple = Tuple[str, type, str]  # (return_key, msg_type, topic)


class ROSNode:
    def __init__(
        self,
        *,
        node_name: str,
        callback: Callable[..., Dict[str, NDArray]],
        subscribes_to: Optional[Sequence[SubTuple]] = None,
        publishes_to: Optional[Sequence[PubTuple]] = None,
        rate_hz: float = 10.0,
        use_single_threaded_executor: bool = True,
        qos_profile: Optional[QoSProfile] = None,
        stale_config: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None:

        # -------- Core config (pure Python, no ROS) --------
        self._node_name = node_name
        self._callback = callback
        self._rate_hz = float(rate_hz)
        if self._rate_hz <= 0:
            raise ValueError("rate_hz must be positive")

        self._use_single = bool(use_single_threaded_executor)

        self._executor: Optional[SingleThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._node: Optional[Node] = None
        self._cb_group: Optional[ReentrantCallbackGroup] = None
        self._timer = None
        self._t0 = None

        # QoS is just stored; actual profile object used once node exists
        self._qos = qos_profile or QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # -------- Staleness (per-arg config only) --------
        # e.g. {"uk": {"after": 0.5, "policy": "zero"}, ...}
        self._stale_cfg = dict(stale_config or {})

        # -------- Normalize subscriptions (config only) --------
        self._subs: List[Tuple[str, type, Callable[[Any], NDArray], str]] = []
        for tup in (subscribes_to or []):
            try:
                topic, msg_type, arg_name = tup
            except Exception:
                raise ValueError(
                    "Each subscription must be (topic_name, msg_type, arg_name)"
                )
            if not isinstance(msg_type, type):
                raise TypeError(f"msg_type for '{tup}' must be a class")
            if not isinstance(arg_name, str) or not arg_name:
                raise TypeError("arg_name must be a non-empty string")

            ros2py = ROS2PY_DEFAULT.get(msg_type)
            if ros2py is None:
                raise TypeError(
                    f"No default ros2py converter for msg_type {msg_type.__name__} "
                    f"on topic '{topic}'. Add one to ROS2PY_DEFAULT."
                )

            self._subs.append((topic, msg_type, ros2py, arg_name))

        sub_topics = [t for (t, _, _, _) in self._subs]
        if len(sub_topics) != len(set(sub_topics)):
            dup = sorted({t for t in sub_topics if sub_topics.count(t) > 1})
            raise ValueError(f"Duplicate subscribe topics: {dup}")

        # -------- Normalize publications (config only) --------
        self._pubs: List[Tuple[str, Callable[[NDArray], Any], type, str]] = []
        for tup in (publishes_to or []):
            try:
                return_key, msg_type, topic = tup
            except Exception:
                raise ValueError(
                    "Each publication must be (return_key, msg_type, topic)"
                )
            if not isinstance(return_key, str) or not return_key:
                raise TypeError("return_key must be a non-empty string")
            if not isinstance(msg_type, type):
                raise TypeError(f"msg_type for {tup} must be a class")

            py2ros = PY2ROS_DEFAULT.get(msg_type)
            if py2ros is None:
                raise TypeError(
                    f"No default py2ros converter for msg_type {msg_type.__name__} "
                    f"on topic '{topic}'. Add one to PY2ROS_DEFAULT."
                )

            self._pubs.append((return_key, py2ros, msg_type, topic))

        pub_topics = [topic for (_, _, _, topic) in self._pubs]
        if len(pub_topics) != len(set(pub_topics)):
            dup = sorted({t for t in pub_topics if pub_topics.count(t) > 1})
            raise ValueError(f"Duplicate publish topics: {dup}")

        # -------- Internal caches (by arg_name) --------
        self._inputs_lock = threading.Lock()
        self._inputs_by_arg: Dict[str, NDArray] = {}
        self._input_time_by_arg: Dict[str, float] = {}

        # Maps that will be filled once we actually create the Node
        self._sub_map: Dict[str, Tuple[type, Callable[[Any], NDArray], str]] = {}
        self._pub_map: Dict[str, Tuple[Callable[[NDArray], Any], type, str, Any]] = {}

    # ----------------------------------------------------------------------
    # NODE CREATION (this is where we actually touch ROS graph)
    # ----------------------------------------------------------------------
    def create_node(self) -> Node:
        """
        Create the underlying rclpy.Node, subscriptions, publishers, and timer.

        This is the moment the object starts participating in the ROS graph.
        Safe to call multiple times; only the first call does the work.
        """
        if self._node is not None:
            return self._node

        if not rclpy.ok():
            rclpy.init()

        node = Node(self._node_name)
        self._node = node
        self._cb_group = ReentrantCallbackGroup()

        # --- Create subscriptions ---
        self._sub_map.clear()
        for topic, msg_type, ros2py, arg_name in self._subs:
            self._sub_map[topic] = (msg_type, ros2py, arg_name)

            node.create_subscription(
                msg_type,
                topic,
                self._make_sub_cb(topic, ros2py, arg_name),
                self._qos,
                callback_group=self._cb_group,
            )

        # --- Create publishers ---
        self._pub_map.clear()
        for (return_key, py2ros, msg_type, topic) in self._pubs:
            pub = node.create_publisher(msg_type, topic, self._qos)
            self._pub_map[return_key] = (py2ros, msg_type, topic, pub)

        # --- Timer for callback ---
        self._t0 = node.get_clock().now()
        self._timer = node.create_timer(
            1.0 / self._rate_hz,
            self._on_timer,
            callback_group=self._cb_group,
        )

        node.get_logger().info(
            f"[{node.get_name()}] subs={list(self._sub_map.keys())} "
            f"pubs={[topic for (_, _, topic, _) in self._pub_map.values()]} "
            f"rate={self._rate_hz} Hz "
        )

        return node

    # --------------------------------------------------------------------------
    # SUBSCRIPTION CALLBACK
    # --------------------------------------------------------------------------
    def _make_sub_cb(self, topic: str, ros2py, arg_name: str):
        def _cb(msg):
            node = self._node
            if node is None:
                return
            try:
                arr = np.asarray(ros2py(msg), dtype=float)
                if arr.ndim == 0:
                    arr = arr[None]
                now = time.monotonic()
                with self._inputs_lock:
                    self._inputs_by_arg[arg_name] = arr
                    self._input_time_by_arg[arg_name] = now
            except Exception as e:
                node.get_logger().error(f"[{topic}] ros2py failed: {e}")
        return _cb

    # --------------------------------------------------------------------------
    # TIMER CALLBACK (runs your system wrapper)
    # --------------------------------------------------------------------------
    def _on_timer(self):
        node = self._node
        if node is None or self._t0 is None:
            return

        tk = (node.get_clock().now() - self._t0).nanoseconds * 1e-9
        now = time.monotonic()

        # Build inputs snapshot
        with self._inputs_lock:
            inputs_snapshot: Dict[str, NDArray] = {}

            for arg_name, arr in self._inputs_by_arg.items():
                cfg = self._stale_cfg.get(arg_name, {})
                after = cfg.get("after", None)     # None = never stale
                policy = cfg.get("policy", "drop") # default action

                is_stale = False
                if after is not None:
                    t_last = self._input_time_by_arg.get(arg_name)
                    if t_last is None or (now - t_last) > float(after):
                        is_stale = True

                if not is_stale:
                    inputs_snapshot[arg_name] = arr.copy()
                else:
                    if policy == "zero":
                        inputs_snapshot[arg_name] = np.zeros_like(arr)
                    elif policy == "hold":
                        inputs_snapshot[arg_name] = arr.copy()
                    elif policy == "drop":
                        continue
                    else:
                        continue

        # Call user callback
        try:
            outputs = self._callback(tk, **inputs_snapshot)
            if not isinstance(outputs, dict):
                return
        except Exception as e:
            node.get_logger().error(f"callback error: {e}")
            return

        # Publish results
        for return_key, (py2ros, _, topic, pub) in self._pub_map.items():
            if return_key not in outputs:
                continue
            try:
                arr = np.asarray(outputs[return_key], dtype=float).ravel()
                msg = py2ros(arr)
                pub.publish(msg)
            except Exception as e:
                node.get_logger().error(f"[{topic}] publish failed: {e}")

    # --------------------------------------------------------------------------
    # CONTROL API
    # --------------------------------------------------------------------------
    def start(self):
        """
        Start spinning the ROS node in a background executor thread.

        If the Node has not yet been created, this will call create_node().
        """
        if self.is_running():
            return

        node = self._node or self.create_node()

        executor = (
            SingleThreadedExecutor() if self._use_single else MultiThreadedExecutor()
        )
        executor.add_node(node)
        self._executor = executor

        self._thread = threading.Thread(
            target=executor.spin,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """
        Stop spinning the executor, but DO NOT destroy the underlying node.
        """
        if self._executor:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def destroy(self):
        """
        Remove the underlying rclpy.Node from the ROS graph.
        """
        self.stop()

        if self._node:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

        self._pub_map.clear()
        self._sub_map.clear()
        self._inputs_by_arg.clear()
        self._input_time_by_arg.clear()
        self._timer = None
        self._t0 = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_inputs(self) -> Dict[str, np.ndarray]:
        """Return a copy of most recent inputs by argument name."""
        with self._inputs_lock:
            return {k: np.asarray(v) for k, v in self._inputs_by_arg.items()}
