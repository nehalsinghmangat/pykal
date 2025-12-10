from typing import Callable, Dict, Optional, Sequence, Tuple, Any, List, Set
import numpy as np
import threading
import time
import inspect
import logging

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from numpy.typing import NDArray

from pykal.utilities.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT

SubTuple = Tuple[str, type, str]  # (topic, msg_type, arg_name)
PubTuple = Tuple[str, type, str]  # (return_key, msg_type, topic)


class ROSNode:
    """
    Wraps arbitrary Python callbacks as ROS2 nodes with robustness features.

    This class provides a high-level interface for creating ROS2 nodes that:
    - Subscribe to topics and convert ROS messages to NumPy arrays
    - Call a user callback at a fixed rate with the latest data
    - Publish callback outputs as ROS messages
    - Handle staleness policies (zero/hold/drop) for multi-rate sensor fusion
    - Validate required topics and callback signatures
    - Track diagnostics (message rates, error counts, uptime)
    - Monitor heartbeat timeouts for connection health
    - Provide error callbacks for graceful degradation

    Parameters
    ----------
    node_name : str
        Name of the ROS2 node
    callback : Callable[..., Dict[str, NDArray]]
        User callback that receives tk (time in seconds) as first argument,
        followed by subscribed topics as kwargs. Must return a dictionary
        mapping return keys to NumPy arrays.
    subscribes_to : Optional[Sequence[SubTuple]]
        List of (topic_name, msg_type, arg_name) tuples for subscriptions.
        The msg_type must have a converter in ROS2PY_DEFAULT.
    publishes_to : Optional[Sequence[PubTuple]]
        List of (return_key, msg_type, topic_name) tuples for publications.
        The msg_type must have a converter in PY2ROS_DEFAULT.
    rate_hz : float, default=10.0
        Callback execution rate in Hz
    use_single_threaded_executor : bool, default=True
        Use SingleThreadedExecutor (True) or MultiThreadedExecutor (False)
    qos_profile : Optional[QoSProfile], default=None
        ROS2 QoS profile for all subscriptions and publishers
    stale_config : Optional[Dict[str, Dict[str, object]]], default=None
        Staleness policies per argument:
        {"arg_name": {"after": 0.5, "policy": "zero"|"hold"|"drop", "use_header_stamp": False}}
        - after: Staleness threshold in seconds
        - policy: What to do with stale data (zero/hold/drop)
        - use_header_stamp: Use msg.header.stamp instead of arrival time (default: False)
    required_topics : Optional[Set[str]], default=None
        Set of arg_names that must be present for callback to execute.
        If any required topic is missing, callback is skipped and error_callback is invoked.
    error_callback : Optional[Callable[[str, Exception], None]], default=None
        Callback invoked on errors: error_callback(context, exception).
        Context strings: "subscription:{topic}", "callback", "publish:{topic}",
        "required_topics", "heartbeat:{arg_name}"
    enable_diagnostics : bool, default=False
        Enable tracking of message counts, rates, errors, latency, and uptime.
        Use get_diagnostics() to retrieve statistics. Latency is automatically
        tracked for messages with header.stamp (difference between message time
        and arrival time).
    heartbeat_config : Optional[Dict[str, float]], default=None
        Heartbeat timeouts per argument: {"arg_name": timeout_seconds}.
        If no message received within timeout, error_callback is invoked with TimeoutError.
    validate_callback_signature : bool, default=True
        Validate callback signature matches subscriptions at initialization.
        Checks that callback has 'tk' as first parameter and accepts all subscription arg_names.

    Examples
    --------
    Basic usage with required topics and diagnostics:

    >>> from geometry_msgs.msg import Twist
    >>> def my_callback(tk, cmd_vel):
    ...     # Process cmd_vel and return output
    ...     return {"output": cmd_vel * 2.0}
    >>>
    >>> node = ROSNode(
    ...     node_name="my_node",
    ...     callback=my_callback,
    ...     subscribes_to=[("/cmd_vel", Twist, "cmd_vel")],
    ...     publishes_to=[("output", Twist, "/output")],
    ...     rate_hz=50.0,
    ...     required_topics={"cmd_vel"},  # Callback won't run without cmd_vel
    ...     enable_diagnostics=True,
    ... )
    >>> node.start()  # doctest: +SKIP
    >>> diag = node.get_diagnostics()  # doctest: +SKIP

    With error callback and heartbeat monitoring:

    >>> def handle_error(context, error):
    ...     print(f"Error in {context}: {error}")
    >>>
    >>> node = ROSNode(
    ...     node_name="robust_node",
    ...     callback=my_callback,
    ...     subscribes_to=[("/sensor", Twist, "sensor_data")],
    ...     error_callback=handle_error,
    ...     heartbeat_config={"sensor_data": 1.0},  # Warn if >1s old
    ... )  # doctest: +SKIP

    See Also
    --------
    DynamicalSystem : Core abstraction for modeling control systems
    """
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
        required_topics: Optional[Set[str]] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None,
        enable_diagnostics: bool = False,
        heartbeat_config: Optional[Dict[str, float]] = None,
        validate_callback_signature: bool = True,
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

        # -------- Robustness features --------
        self._required_topics = set(required_topics or set())
        self._error_callback = error_callback
        self._enable_diagnostics = bool(enable_diagnostics)
        self._validate_signature = bool(validate_callback_signature)

        # Heartbeat config: {"arg_name": timeout_seconds}
        self._heartbeat_cfg = dict(heartbeat_config or {})

        # Diagnostics tracking
        self._diag_lock = threading.Lock()
        self._diag_msg_count: Dict[str, int] = {}
        self._diag_error_count: Dict[str, int] = {}
        self._diag_last_msg_time: Dict[str, float] = {}
        self._diag_latency_sum: Dict[str, float] = {}  # For average latency
        self._diag_latency_count: Dict[str, int] = {}  # Number of latency samples
        self._diag_callback_count = 0
        self._diag_callback_errors = 0
        self._diag_start_time: Optional[float] = None

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

        # -------- Validate callback signature --------
        if self._validate_signature:
            self._validate_callback_signature()

        # -------- Validate required topics --------
        if self._required_topics:
            sub_arg_names = {arg_name for (_, _, _, arg_name) in self._subs}
            missing = self._required_topics - sub_arg_names
            if missing:
                raise ValueError(
                    f"Required topics {missing} not found in subscriptions. "
                    f"Available: {sub_arg_names}"
                )

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
        self._input_time_by_arg: Dict[str, float] = {}  # Arrival time (monotonic)
        self._input_msg_time_by_arg: Dict[str, float] = {}  # Message time (header.stamp)

        # Maps that will be filled once we actually create the Node
        self._sub_map: Dict[str, Tuple[type, Callable[[Any], NDArray], str]] = {}
        self._pub_map: Dict[str, Tuple[Callable[[NDArray], Any], type, str, Any]] = {}

    # ----------------------------------------------------------------------
    # VALIDATION & DIAGNOSTICS HELPERS
    # ----------------------------------------------------------------------
    def _validate_callback_signature(self) -> None:
        """Validate that callback signature matches subscriptions."""
        try:
            sig = inspect.signature(self._callback)
        except Exception:
            return  # Can't introspect, skip validation

        params = list(sig.parameters.keys())
        if not params or params[0] != "tk":
            raise ValueError(
                "Callback must have 'tk' as first parameter. "
                f"Got: {params[0] if params else 'no parameters'}"
            )

        expected_args = {"tk"} | {arg_name for (_, _, _, arg_name) in self._subs}
        actual_args = set(params)

        # Check for VAR_KEYWORD (**kwargs)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if has_var_keyword:
            return  # **kwargs allows any additional arguments

        # Otherwise, check argument compatibility
        unexpected = actual_args - expected_args
        if unexpected:
            raise ValueError(
                f"Callback has unexpected parameters: {unexpected}. "
                f"Expected: {expected_args}"
            )

    def _call_error_callback(self, context: str, error: Exception) -> None:
        """Call user error callback if configured."""
        if self._error_callback is not None:
            try:
                self._error_callback(context, error)
            except Exception as e:
                if self._node:
                    self._node.get_logger().error(
                        f"Error callback failed for {context}: {e}"
                    )

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic statistics about the node's operation.

        Returns:
            Dictionary containing:
            - uptime_seconds: Time since node started
            - callback_count: Number of timer callbacks executed
            - callback_errors: Number of callback errors
            - callback_rate_hz: Average callback rate
            - topics: Per-topic statistics (msg_count, error_count, rate_hz, last_msg_age, latency_ms)
        """
        with self._diag_lock:
            now = time.monotonic()
            uptime = (
                now - self._diag_start_time if self._diag_start_time is not None else 0.0
            )

            topics = {}
            for arg_name in self._diag_msg_count.keys():
                msg_count = self._diag_msg_count.get(arg_name, 0)
                error_count = self._diag_error_count.get(arg_name, 0)
                last_msg_time = self._diag_last_msg_time.get(arg_name)

                # Calculate average latency if available
                latency_sum = self._diag_latency_sum.get(arg_name, 0.0)
                latency_count = self._diag_latency_count.get(arg_name, 0)
                avg_latency_ms = (
                    (latency_sum / latency_count) * 1000.0 if latency_count > 0 else None
                )

                topics[arg_name] = {
                    "msg_count": msg_count,
                    "error_count": error_count,
                    "rate_hz": msg_count / uptime if uptime > 0 else 0.0,
                    "last_msg_age": (
                        now - last_msg_time if last_msg_time is not None else None
                    ),
                    "latency_ms": avg_latency_ms,
                }

            return {
                "uptime_seconds": uptime,
                "callback_count": self._diag_callback_count,
                "callback_errors": self._diag_callback_errors,
                "callback_rate_hz": (
                    self._diag_callback_count / uptime if uptime > 0 else 0.0
                ),
                "topics": topics,
            }

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

        # Initialize diagnostics
        if self._enable_diagnostics:
            with self._diag_lock:
                self._diag_start_time = time.monotonic()
                for _, _, _, arg_name in self._subs:
                    self._diag_msg_count[arg_name] = 0
                    self._diag_error_count[arg_name] = 0
                    self._diag_latency_sum[arg_name] = 0.0
                    self._diag_latency_count[arg_name] = 0

        node.get_logger().info(
            f"[{node.get_name()}] subs={list(self._sub_map.keys())} "
            f"pubs={[topic for (_, _, topic, _) in self._pub_map.values()]} "
            f"rate={self._rate_hz} Hz "
            f"required={list(self._required_topics) if self._required_topics else 'none'} "
            f"diagnostics={'enabled' if self._enable_diagnostics else 'disabled'}"
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

                # Extract message timestamp if available (header.stamp)
                msg_time = None
                if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                    try:
                        # Convert ROS time to seconds
                        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    except Exception:
                        pass  # If extraction fails, msg_time stays None

                with self._inputs_lock:
                    self._inputs_by_arg[arg_name] = arr
                    self._input_time_by_arg[arg_name] = now
                    if msg_time is not None:
                        self._input_msg_time_by_arg[arg_name] = msg_time

                # Update diagnostics
                if self._enable_diagnostics:
                    with self._diag_lock:
                        self._diag_msg_count[arg_name] = (
                            self._diag_msg_count.get(arg_name, 0) + 1
                        )
                        self._diag_last_msg_time[arg_name] = now

                        # Track latency if message timestamp available
                        if msg_time is not None:
                            # Calculate latency (arrival time - message time)
                            # Note: This assumes system clocks are synchronized
                            ros_now = node.get_clock().now()
                            ros_now_sec = ros_now.nanoseconds * 1e-9
                            latency = ros_now_sec - msg_time

                            # Only track positive latencies (ignore clock sync issues)
                            if latency > 0 and latency < 10.0:  # Sanity check: < 10s
                                self._diag_latency_sum[arg_name] = (
                                    self._diag_latency_sum.get(arg_name, 0.0) + latency
                                )
                                self._diag_latency_count[arg_name] = (
                                    self._diag_latency_count.get(arg_name, 0) + 1
                                )

            except Exception as e:
                node.get_logger().error(f"[{topic}] ros2py failed: {e}")

                # Track error in diagnostics
                if self._enable_diagnostics:
                    with self._diag_lock:
                        self._diag_error_count[arg_name] = (
                            self._diag_error_count.get(arg_name, 0) + 1
                        )

                # Call user error callback
                self._call_error_callback(f"subscription:{topic}", e)

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
                use_header_stamp = cfg.get("use_header_stamp", False)

                is_stale = False
                if after is not None:
                    if use_header_stamp and arg_name in self._input_msg_time_by_arg:
                        # Use message timestamp (header.stamp) for staleness check
                        msg_time = self._input_msg_time_by_arg[arg_name]
                        ros_now = node.get_clock().now()
                        ros_now_sec = ros_now.nanoseconds * 1e-9
                        age = ros_now_sec - msg_time
                        if age > float(after):
                            is_stale = True
                    else:
                        # Use arrival time (default behavior)
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

        # Check required topics are present
        if self._required_topics:
            missing = self._required_topics - set(inputs_snapshot.keys())
            if missing:
                err = ValueError(f"Required topics missing: {missing}")
                node.get_logger().warn(f"Skipping callback: {err}")
                self._call_error_callback("required_topics", err)
                return

        # Check heartbeat timeouts
        if self._heartbeat_cfg:
            for arg_name, timeout in self._heartbeat_cfg.items():
                if arg_name in self._input_time_by_arg:
                    t_last = self._input_time_by_arg[arg_name]
                    age = now - t_last
                    if age > timeout:
                        err = TimeoutError(
                            f"Heartbeat timeout for '{arg_name}': "
                            f"{age:.2f}s > {timeout:.2f}s"
                        )
                        node.get_logger().warn(str(err))
                        self._call_error_callback(f"heartbeat:{arg_name}", err)

        # Call user callback
        try:
            outputs = self._callback(tk, **inputs_snapshot)
            if not isinstance(outputs, dict):
                return

            # Track successful callback
            if self._enable_diagnostics:
                with self._diag_lock:
                    self._diag_callback_count += 1

        except Exception as e:
            node.get_logger().error(f"callback error: {e}")

            # Track callback error
            if self._enable_diagnostics:
                with self._diag_lock:
                    self._diag_callback_errors += 1

            # Call user error callback
            self._call_error_callback("callback", e)
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
                self._call_error_callback(f"publish:{topic}", e)

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
