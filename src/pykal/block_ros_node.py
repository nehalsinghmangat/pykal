from typing import Callable, Dict, Optional, Sequence, Tuple, Any, List, Literal
import numpy as np
import threading
import time  # <-- for staleness timing

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from numpy.typing import NDArray

from std_msgs.msg import Float64MultiArray


# -------------------- Default converters --------------------

def _default_ros2py(msg: Any) -> np.ndarray:
    """Default ROS->NumPy converter for Float64MultiArray."""
    if isinstance(msg, Float64MultiArray):
        return np.asarray(msg.data, dtype=float)
    raise TypeError(
        f"No default converter for message type {type(msg).__name__}. "
        "Provide a ros2py converter for this subscription."
    )

def _default_py2ros(arr: np.ndarray) -> Any:
    """Default NumPy->ROS converter to Float64MultiArray."""
    msg = Float64MultiArray()
    msg.data = np.asarray(arr, dtype=float).ravel().tolist()
    return msg


# ======================== BlockROSNode ========================

class BlockROSNode(Node):
    """
    Tuple-based I/O declaration:

      subscribes=[
        (topic_name, msg_type, ros2py, arg_name),
        ...
      ]

      publishes=[
        (return_key, py2ros, msg_type, topic_name),
        ...
      ]

    On each timer tick:
        inputs_by_arg = {arg_name: np.ndarray, ...}
        outputs = system_wrapper(tk, **inputs_by_arg)
    And publish outputs[return_key] to the configured topic.

    Staleness handling (per input):
      - Pass global defaults via `stale_after_sec` (float|None) and `stale_policy` ('drop'|'zero'|'hold').
      - Override per argument with `stale_config = {arg_name: {'after': float|None, 'policy': 'drop'|'zero'|'hold'}}`.
      - 'drop' = omit the arg when stale, 'zero' = zeros_like(last_value), 'hold' = keep last value.
      - If an arg has never arrived before, it is simply absent (cannot zero without a shape).

    Notes
    -----
    - `ros2py` / `py2ros` can be None to use defaults (only for Float64MultiArray).
    - Inputs are cached by *argument name* (decoupled from topic name).
    """

    def __init__(
        self,
        node_name: str,
        system_wrapper: Callable[..., Dict[str, NDArray]],  # outputs dict
        *,
        subscribes: Optional[Sequence[Tuple[str, type, Optional[Callable[[Any], NDArray]], str]]] = None,
        publishes: Optional[Sequence[Tuple[str, Optional[Callable[[NDArray], Any]], type, str]]] = None,
        rate_hz: float = 10.0,
        use_single_threaded_executor: bool = True,
        shutdown_on_stop: bool = True,
        qos_profile: Optional[QoSProfile] = None,
        # -------- staleness controls ----------
        stale_after_sec: Optional[float] = None,
        stale_policy: Literal["drop", "zero", "hold"] = "drop",
        stale_config: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None:
        # Do NOT init rclpy here; do it in start()
        super().__init__(node_name)

        # Core
        self._system_wrapper = system_wrapper
        self._rate_hz = float(rate_hz)
        if self._rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self._use_single = bool(use_single_threaded_executor)
        self._shutdown_on_stop = bool(shutdown_on_stop)

        self._executor: Optional[SingleThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None

        # ROS infra
        self._qos = qos_profile or QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._cb_group = ReentrantCallbackGroup()

        # ---------- Staleness config ----------
        self._stale_after_default: Optional[float] = float(stale_after_sec) if stale_after_sec is not None else None
        self._stale_policy_default: Literal["drop", "zero", "hold"] = stale_policy
        # Example override: {"uk": {"after": 1.0, "policy": "zero"}, "yk": {"after": None, "policy": "hold"}}
        self._stale_cfg: Dict[str, Dict[str, object]] = dict(stale_config or {})

        # ---------- Normalize subscription tuples ----------
        self._subs: List[Tuple[str, type, Optional[Callable[[Any], NDArray]], str]] = []
        for tup in (subscribes or []):
            try:
                topic, msg_type, ros2py, arg_name = tup
            except Exception:
                raise ValueError(
                    "Each subscription must be (topic_name, msg_type, ros2py_converter, arg_name_for_system_wrapper)"
                )
            if not isinstance(msg_type, type):
                raise TypeError(f"Subscription msg_type for '{tup}' must be a class")
            if ros2py is not None and not callable(ros2py):
                raise TypeError(f"ros2py converter for '{topic}' must be callable (or None for default)")
            if not isinstance(arg_name, str) or not arg_name:
                raise TypeError(f"arg_name must be a non-empty string for topic '{topic}'")
            self._subs.append((topic, msg_type, ros2py, arg_name))

        # No duplicate sub topics
        sub_topics = [t for (t, _, _, _) in self._subs]
        if len(set(sub_topics)) != len(sub_topics):
            dup = sorted({t for t in sub_topics if sub_topics.count(t) > 1})
            raise ValueError(f"Duplicate subscribe topics: {dup}")

        # ---------- Normalize publication tuples ----------
        self._pubs: List[Tuple[str, Optional[Callable[[NDArray], Any]], type, str]] = []
        for tup in (publishes or []):
            try:
                return_key, py2ros, msg_type, topic = tup
            except Exception:
                raise ValueError("Each publication must be (return_key, py2ros_converter, msg_type, topic_name)")
            if not isinstance(return_key, str) or not return_key:
                raise TypeError("return_key must be a non-empty string")
            if not isinstance(msg_type, type):
                raise TypeError(f"Publication msg_type for '{tup}' must be a class")
            if py2ros is not None and not callable(py2ros):
                raise TypeError(f"py2ros converter for '{topic}' must be callable (or None for default)")
            self._pubs.append((return_key, py2ros, msg_type, topic))

        # No duplicate pub topics
        pub_topics = [topic for (_, _, _, topic) in self._pubs]
        if len(set(pub_topics)) != len(pub_topics):
            dup = sorted({t for t in pub_topics if pub_topics.count(t) > 1})
            raise ValueError(f"Duplicate publish topics: {dup}")

        # ---------- Internal maps ----------
        # Cache latest inputs by *arg name* (not by topic)
        self._inputs_lock = threading.Lock()
        self._inputs_by_arg: Dict[str, NDArray] = {}
        self._input_time_by_arg: Dict[str, float] = {}  # last arrival time per arg_name

        # topic -> (msg_type, ros2py, arg_name)
        self._sub_map: Dict[str, Tuple[type, Callable[[Any], NDArray], str]] = {}
        for (topic, msg_type, ros2py, arg_name) in self._subs:
            # Fill default converter if None (only for Float64MultiArray)
            if ros2py is None:
                if msg_type is Float64MultiArray:
                    ros2py = _default_ros2py
                else:
                    raise TypeError(
                        f"No ros2py provided for topic '{topic}' and no default for {msg_type.__name__}"
                    )
            self._sub_map[topic] = (msg_type, ros2py, arg_name)

        # return_key -> (py2ros, msg_type, topic, publisher)
        self._pub_map: Dict[str, Tuple[Callable[[NDArray], Any], type, str, Any]] = {}
        for (return_key, py2ros, msg_type, topic) in self._pubs:
            if py2ros is None:
                if msg_type is Float64MultiArray:
                    py2ros = _default_py2ros
                else:
                    raise TypeError(
                        f"No py2ros provided for topic '{topic}' and no default for {msg_type.__name__}"
                    )
            pub = self.create_publisher(msg_type, topic, self._qos)
            self._pub_map[return_key] = (py2ros, msg_type, topic, pub)

        # ---------- Subscriptions ----------
        for topic, (msg_type, ros2py, arg_name) in self._sub_map.items():
            self.create_subscription(
                msg_type, topic, self._make_sub_cb(topic, ros2py, arg_name),
                self._qos, callback_group=self._cb_group
            )

        # ---------- Timer ----------
        self._t0 = self.get_clock().now()
        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer, callback_group=self._cb_group)

        self.get_logger().info(
            f"[{self.get_name()}] subs={list(self._sub_map.keys())} "
            f"pubs={[topic for (_, _, topic, _) in self._pub_map.values()]} "
            f"rate={self._rate_hz} Hz; stale_default(after={self._stale_after_default}, policy={self._stale_policy_default})"
        )

    # ------------------------------ Callbacks ---------------------------------

    def _make_sub_cb(self, topic: str, ros2py: Callable[[Any], NDArray], arg_name: str):
        def _cb(msg):
            try:
                arr = np.asarray(ros2py(msg), dtype=float)
                if arr.ndim == 0:
                    arr = arr[None]
                now = time.monotonic()  # timestamp arrival
                with self._inputs_lock:
                    self._inputs_by_arg[arg_name] = arr
                    self._input_time_by_arg[arg_name] = now
            except Exception as e:
                self.get_logger().error(f"[{topic}] ros2py conversion failed: {e}")
        return _cb

    def _on_timer(self):
        # elapsed ROS time (seconds)
        tk = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        now = time.monotonic()

        # snapshot inputs (by arg name) with staleness policy applied
        with self._inputs_lock:
            inputs_snapshot: Dict[str, NDArray] = {}
            for arg_name, arr in self._inputs_by_arg.items():
                # resolve per-arg settings with fallbacks
                cfg = self._stale_cfg.get(arg_name, {})
                after = cfg.get("after", self._stale_after_default)
                policy = cfg.get("policy", self._stale_policy_default)

                # None => never stale
                is_stale = False
                if after is not None:
                    t_last = self._input_time_by_arg.get(arg_name, None)
                    if (t_last is None) or ((now - t_last) > float(after)):
                        is_stale = True

                if not is_stale:
                    inputs_snapshot[arg_name] = arr.copy()
                else:
                    if policy == "zero":
                        # Only possible if we've received at least one value (we have a shape)
                        inputs_snapshot[arg_name] = np.zeros_like(arr)
                    elif policy == "hold":
                        inputs_snapshot[arg_name] = arr.copy()
                    elif policy == "drop":
                        # omit this arg entirely
                        continue
                    else:
                        # unknown policy -> default to drop
                        continue

        # Optional: warn if some required args haven't arrived yet
        # (Required = args from subscriptions; some may be missing if never received or dropped.)
        required = {arg for (_, _, _, arg) in self._subs}
        missing = sorted(arg for arg in required if arg not in inputs_snapshot)
        if missing:
            self.get_logger().debug(f"Missing/stale inputs {missing} at tk={tk:.3f}")

        # run system wrapper as system_wrapper(tk, **inputs)
        try:
            outputs = self._system_wrapper(tk, **inputs_snapshot)
            if not outputs:
                return
            if not isinstance(outputs, dict):
                self.get_logger().warning("system_wrapper did not return a dict; ignoring.")
                return
        except Exception as e:
            self.get_logger().error(f"system_wrapper error: {e}")
            return

        # publish only declared return_keys
        for return_key, (py2ros, _msg_type, topic, pub) in self._pub_map.items():
            if return_key not in outputs:
                continue
            try:
                arr = np.asarray(outputs[return_key], dtype=float).ravel()
                msg = py2ros(arr)
                pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"[{topic}] py2ros publish failed: {e}")

    # ---------------------------- Public API ----------------------------------

    def start(self):
        """Start the ROS 2 executor in a background thread and spin this node."""
        if self.is_running():
            return
        if not rclpy.ok():
            rclpy.init(args=None)
        executor = SingleThreadedExecutor() if self._use_single else MultiThreadedExecutor()
        executor.add_node(self)
        self._executor = executor
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        print(
            f"[ROSBlock] started node='{self.get_name()}' "
            f"pub={[topic for (_, _, _, topic) in self._pubs]} "
            f"sub={[topic for (topic, _, _, _) in self._subs]} @ {self._rate_hz} Hz"
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

    def is_running(self) -> bool:
        """True if executor thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def node(self):
        """Return the underlying Node (self)."""
        return self

    def latest_inputs(self) -> Dict[str, np.ndarray]:
        """Return most recent inputs by system arg name (copy)."""
        with self._inputs_lock:
            return {k: np.asarray(v) for k, v in self._inputs_by_arg.items()}
