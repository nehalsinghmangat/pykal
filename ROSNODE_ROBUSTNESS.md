# ROSNode Robustness Features

This document describes the robustness enhancements added to `pykal.ROSNode` for production-ready robotics applications.

## Overview

The enhanced ROSNode provides seven key robustness features:

1. **Required Topics** - Ensure critical data is present before running control callbacks
2. **Error Callbacks** - Graceful error handling and degradation
3. **Diagnostics** - Real-time monitoring of message rates, errors, latency, and uptime
4. **Heartbeat Monitoring** - Detect connection failures and stale data
5. **Signature Validation** - Catch configuration errors at initialization
6. **Message Timestamp Staleness** - Use sensor timestamps (header.stamp) for staleness detection
7. **Latency Tracking** - Automatically track message delays from sensor to arrival

All features are **backward compatible** - existing code continues to work unchanged.

## 1. Required Topics

Mark certain topics as critical for system operation. If required topics are missing, the callback is skipped and an error is reported.

```python
from pykal import ROSNode
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

def controller(tk, cmd_vel, lidar):
    """Control callback - requires both inputs."""
    # Safe to assume both cmd_vel and lidar are present
    return {"control_output": cmd_vel * 0.5}

node = ROSNode(
    node_name="safe_controller",
    callback=controller,
    subscribes_to=[
        ("/cmd_vel", Twist, "cmd_vel"),
        ("/scan", LaserScan, "lidar"),
    ],
    required_topics={"cmd_vel", "lidar"},  # Both required!
)
```

**When to use**: Systems where missing critical sensor data should halt control (e.g., obstacle avoidance requires lidar).

## 2. Error Callbacks

Get notified when errors occur anywhere in the ROS node pipeline, enabling graceful degradation and logging.

```python
def error_handler(context: str, error: Exception):
    """Called on any error in the node."""
    print(f"[{context}] {type(error).__name__}: {error}")

    # Take action based on context
    if context.startswith("subscription:"):
        # Log sensor failure
        pass
    elif context == "callback":
        # Switch to safe mode
        pass
    elif context.startswith("heartbeat:"):
        # Alert operator
        pass

node = ROSNode(
    node_name="robust_node",
    callback=my_callback,
    subscribes_to=[...],
    error_callback=error_handler,
)
```

**Error contexts**:
- `"subscription:{topic}"` - Message conversion failed
- `"callback"` - User callback raised exception
- `"publish:{topic}"` - Publishing failed
- `"required_topics"` - Required topic missing
- `"heartbeat:{arg_name}"` - Heartbeat timeout exceeded

## 3. Diagnostics

Track real-time statistics about node operation for monitoring, tuning, and debugging.

```python
node = ROSNode(
    node_name="monitored_node",
    callback=my_callback,
    subscribes_to=[
        ("/sensor1", Twist, "s1"),
        ("/sensor2", Twist, "s2"),
    ],
    enable_diagnostics=True,  # Enable tracking
)

node.start()

# Later, check statistics
diag = node.get_diagnostics()

print(f"Uptime: {diag['uptime_seconds']:.1f}s")
print(f"Callbacks: {diag['callback_count']} ({diag['callback_rate_hz']:.1f} Hz)")
print(f"Errors: {diag['callback_errors']}")

for topic, stats in diag['topics'].items():
    print(f"\n{topic}:")
    print(f"  Messages: {stats['msg_count']}")
    print(f"  Rate: {stats['rate_hz']:.1f} Hz")
    print(f"  Errors: {stats['error_count']}")
    print(f"  Age: {stats['last_msg_age']:.3f}s")
```

**Diagnostic fields**:
- `uptime_seconds` - Time since node started
- `callback_count` - Number of timer callbacks executed
- `callback_errors` - Number of callback exceptions
- `callback_rate_hz` - Average callback execution rate
- `topics` - Per-topic statistics:
  - `msg_count` - Messages received
  - `error_count` - Conversion errors
  - `rate_hz` - Average message rate
  - `last_msg_age` - Seconds since last message

**When to use**: Production systems requiring monitoring, performance tuning, or anomaly detection.

## 4. Heartbeat Monitoring

Detect when topics stop publishing or become stale, even if staleness policies keep the system running.

```python
def heartbeat_error_handler(context: str, error: Exception):
    if context.startswith("heartbeat:"):
        # Extract which sensor timed out
        sensor = context.split(":")[1]
        print(f"WARNING: {sensor} heartbeat timeout!")
        # Alert operator, switch to backup sensor, etc.

node = ROSNode(
    node_name="heartbeat_monitor",
    callback=sensor_fusion,
    subscribes_to=[
        ("/imu", Imu, "imu"),
        ("/gps", NavSatFix, "gps"),
    ],
    heartbeat_config={
        "imu": 0.1,   # Warn if IMU silent for >100ms
        "gps": 2.0,   # Warn if GPS silent for >2s
    },
    error_callback=heartbeat_error_handler,
)
```

**Difference from staleness policies**:
- **Staleness policies** (zero/hold/drop) handle what to do with stale data
- **Heartbeat monitoring** detects and alerts when connections are lost

Use both together for robust multi-rate sensor fusion.

## 5. Signature Validation

Catch configuration errors at initialization by validating that your callback signature matches the subscribed topics.

```python
def my_callback(tk, sensor_a, sensor_b):
    """Expects sensor_a and sensor_b."""
    return {"output": sensor_a + sensor_b}

# This will FAIL at initialization with clear error message
try:
    node = ROSNode(
        node_name="misconfigured",
        callback=my_callback,
        subscribes_to=[
            ("/sensor_x", Twist, "sensor_x"),  # Wrong! Callback expects sensor_a
        ],
        validate_callback_signature=True,  # Default
    )
except ValueError as e:
    print(f"Configuration error caught early: {e}")
    # "Callback has unexpected parameters: {'sensor_a', 'sensor_b'}. Expected: {'tk', 'sensor_x'}"
```

**Validation rules**:
1. First parameter must be `tk` (time)
2. Remaining parameters must match subscription `arg_name` values
3. Callbacks with `**kwargs` bypass validation (accept anything)

**Disable if needed**:
```python
node = ROSNode(
    ...
    validate_callback_signature=False,  # Disable for dynamic callbacks
)
```

## 6. Message Timestamp Staleness

Use message timestamps (`header.stamp`) instead of arrival time for staleness detection. This is useful when sensors have synchronized timestamps and you want staleness based on sensor time rather than network delay.

```python
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

def sensor_fusion(tk, imu, gps):
    """Fuse IMU and GPS data."""
    return {"estimate": fuse(imu, gps)}

node = ROSNode(
    node_name="timestamp_fusion",
    callback=sensor_fusion,
    subscribes_to=[
        ("/imu/data", Imu, "imu"),
        ("/gps/fix", Odometry, "gps"),
    ],
    stale_config={
        "imu": {
            "after": 0.05,               # 50ms staleness threshold
            "policy": "hold",
            "use_header_stamp": True     # Use msg.header.stamp for staleness
        },
        "gps": {
            "after": 1.0,                # 1s staleness threshold
            "policy": "zero",
            "use_header_stamp": False    # Use arrival time (default)
        }
    },
)
```

**Key differences**:
- `use_header_stamp=True` - Staleness based on sensor time (`msg.header.stamp`)
- `use_header_stamp=False` (default) - Staleness based on arrival time (network + processing delay)

**When to use**:
- Sensors with hardware-synchronized timestamps (IMU, cameras with triggering)
- Multi-robot systems with synchronized clocks
- Playback from rosbags where you want deterministic behavior

**When NOT to use**:
- Message types without `header.stamp` (falls back to arrival time automatically)
- Clocks not synchronized (will give incorrect staleness)
- Real-time systems where arrival time matters more than sensor time

## 7. Latency Tracking

Automatically track message latency (delay between sensor time and arrival time) for all topics with `header.stamp`. Useful for diagnosing network delays, processing bottlenecks, and system health.

```python
node = ROSNode(
    node_name="latency_monitor",
    callback=my_callback,
    subscribes_to=[
        ("/camera/image", Image, "image"),
        ("/lidar/scan", LaserScan, "scan"),
    ],
    enable_diagnostics=True,  # Enable latency tracking
)

node.start()

# Check latency periodically
import time
while True:
    time.sleep(5.0)
    diag = node.get_diagnostics()

    for topic, stats in diag['topics'].items():
        latency = stats.get('latency_ms')
        if latency is not None:
            print(f"{topic}: {latency:.1f} ms latency")
        else:
            print(f"{topic}: No header.stamp (latency not available)")

        # Alert if latency too high
        if latency and latency > 100.0:
            print(f"  WARNING: High latency on {topic}!")
```

**Latency calculation**:
- Latency = arrival time (`node.get_clock().now()`) - message time (`msg.header.stamp`)
- Reported as average over all messages
- Only meaningful when clocks are synchronized

**Diagnostic output**:
```python
diag = node.get_diagnostics()
# {
#     'uptime_seconds': 120.5,
#     'callback_count': 1205,
#     'topics': {
#         'imu': {
#             'msg_count': 12050,
#             'rate_hz': 100.0,
#             'latency_ms': 12.3,    # NEW: Average latency
#             'last_msg_age': 0.01,
#         }
#     }
# }
```

**When to use**:
- Production monitoring - detect network/processing issues
- Performance tuning - identify bottlenecks
- Health checks - alert when latency exceeds thresholds
- Debugging - understand system timing behavior

**Sanity checks**:
- Only positive latencies tracked (ignores clock sync issues)
- Latencies > 10 seconds ignored (likely clock desync)
- Returns `None` for topics without `header.stamp`

## Complete Example: Production-Ready Multi-Sensor Fusion

```python
from pykal import ROSNode
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

# Track system health
system_errors = []

def error_handler(context: str, error: Exception):
    """Log all errors for analysis."""
    system_errors.append({
        "time": time.time(),
        "context": context,
        "error": str(error),
    })

    # Critical errors trigger safe mode
    if context == "required_topics":
        engage_safe_mode()
    elif context.startswith("heartbeat:imu"):
        switch_to_backup_imu()

def sensor_fusion(tk, imu, odom, cmd_vel=None):
    """Fuse IMU and odometry, optionally use cmd_vel."""
    # IMU and odom are required (guaranteed present)
    # cmd_vel is optional (uses staleness policy)

    estimate = fuse_imu_odom(imu, odom)

    if cmd_vel is not None:
        estimate = refine_with_velocity(estimate, cmd_vel)

    return {"state_estimate": estimate}

# Create robust sensor fusion node
fusion_node = ROSNode(
    node_name="robust_sensor_fusion",
    callback=sensor_fusion,
    subscribes_to=[
        ("/imu/data", Imu, "imu"),
        ("/odom", Odometry, "odom"),
        ("/cmd_vel", Twist, "cmd_vel"),
    ],
    publishes_to=[
        ("state_estimate", Odometry, "/state_estimate"),
    ],
    rate_hz=100.0,  # 100 Hz control loop

    # Robustness features
    required_topics={"imu", "odom"},  # Critical sensors
    error_callback=error_handler,     # Track all errors
    enable_diagnostics=True,          # Monitor performance
    heartbeat_config={
        "imu": 0.05,   # IMU must update at >20 Hz
        "odom": 0.1,   # Odom must update at >10 Hz
    },
    stale_config={
        "imu": {
            "after": 0.05,
            "policy": "hold",
            "use_header_stamp": True,  # Use sensor timestamp
        },
        "cmd_vel": {
            "after": 0.5,
            "policy": "zero",
            "use_header_stamp": False,  # Use arrival time
        },
    },
    validate_callback_signature=True,
)

fusion_node.start()

# Monitor system health
import time
while True:
    time.sleep(5.0)
    diag = fusion_node.get_diagnostics()

    print(f"\n=== System Health ({diag['uptime_seconds']:.0f}s) ===")
    print(f"Control rate: {diag['callback_rate_hz']:.1f} Hz")
    print(f"Errors: {len(system_errors)} total")

    for topic, stats in diag['topics'].items():
        status = "✓" if stats['last_msg_age'] < 1.0 else "⚠"
        latency = stats.get('latency_ms')
        latency_str = f", {latency:.1f}ms latency" if latency else ""
        print(f"{status} {topic}: {stats['rate_hz']:.1f} Hz, {stats['error_count']} errors{latency_str}")
```

## Migration Guide

Existing code requires no changes:

```python
# Old code still works
node = ROSNode(
    node_name="legacy_node",
    callback=my_callback,
    subscribes_to=[...],
)
```

Add robustness features incrementally:

```python
# Add diagnostics first
node = ROSNode(
    ...
    enable_diagnostics=True,
)

# Then add error handling
node = ROSNode(
    ...
    enable_diagnostics=True,
    error_callback=my_error_handler,
)

# Finally, add required topics and heartbeat
node = ROSNode(
    ...
    enable_diagnostics=True,
    error_callback=my_error_handler,
    required_topics={"critical_sensor"},
    heartbeat_config={"critical_sensor": 1.0},
)
```

## Best Practices

1. **Start with diagnostics** - Enable on all nodes to understand baseline performance and track latency
2. **Use required_topics for safety-critical systems** - Don't run control without critical sensors
3. **Combine heartbeat + staleness** - Heartbeat detects failures, staleness handles multi-rate data
4. **Error callbacks for graceful degradation** - Don't crash, switch to safe mode
5. **Validate signatures in development** - Catch config errors early, disable in production if needed
6. **Use header.stamp for synchronized sensors** - IMU, cameras, and time-critical sensors benefit from message timestamp staleness
7. **Monitor latency in production** - Track message delays to detect network issues and processing bottlenecks early

## Performance

All features have minimal overhead:
- Diagnostics (including latency): ~2-3 µs per message (only when enabled)
- Signature validation: One-time at initialization
- Error callbacks: Only invoked on errors
- Required topics check: ~1 µs per callback (only when configured)
- Heartbeat: ~1 µs per monitored topic per callback
- Message timestamp staleness: ~0.5 µs per message (only when use_header_stamp=True)
- Header.stamp extraction: ~0.3 µs per message (automatic for messages with headers)

Disable features you don't need for maximum performance. Latency tracking has negligible overhead when diagnostics are already enabled.
