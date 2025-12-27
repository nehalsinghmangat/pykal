:doc:`← Python to ROS <index>`

====================================
Working with Custom ROS Message Types
====================================

By default, ``pykal`` supports common ROS2 message types (Twist, Odometry, Vector3, etc.). This guide shows how to use **custom message types** with ``pykal``'s ROSNode wrapper.

.. contents:: Quick Navigation
   :local:
   :depth: 2

Why Custom Messages?
====================

Use custom messages when:

- Standard messages don't fit your data structure
- You need specific field names for clarity
- You're interfacing with existing ROS packages that use custom types
- You want type safety for complex data structures

**Example Use Cases**:

- ``EstimateWithCovariance`` - State estimate + uncertainty
- ``ControlCommand`` - Multi-DOF control with metadata
- ``SensorFusion`` - Combined data from multiple sensors
- ``RobotStatus`` - Custom telemetry fields

Message Converter Architecture
===============================

``pykal`` uses two registries for bidirectional conversion:

.. code-block:: python

   from pykal.ros.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT
   
   # ROS → NumPy (for subscribers)
   ROS2PY_DEFAULT[MessageType] = converter_function
   
   # NumPy → ROS (for publishers)
   PY2ROS_DEFAULT[MessageType] = converter_function

**Conversion Flow**:

::

    Subscriber: ROS Topic → ROS Message → ROS2PY → NumPy Array → Callback
    Publisher:  Callback → NumPy Array → PY2ROS → ROS Message → ROS Topic

Creating a Custom Message
==========================

Step 1: Define Message in Package
----------------------------------

Create a ROS2 package with custom messages:

.. code-block:: bash

   # Create package
   ros2 pkg create --build-type ament_cmake my_custom_msgs
   
   cd my_custom_msgs
   mkdir msg

Create message definition file:

.. code-block:: text

   # msg/EstimateWithCovariance.msg
   float64[] state         # State estimate vector
   float64[] covariance    # Flattened covariance matrix
   uint32 state_dim        # State dimension
   builtin_interfaces/Time stamp

Step 2: Configure Package
--------------------------

Update ``CMakeLists.txt``:

.. code-block:: cmake

   find_package(rosidl_default_generators REQUIRED)
   find_package(builtin_interfaces REQUIRED)
   
   rosidl_generate_interfaces(${PROJECT_NAME}
     "msg/EstimateWithCovariance.msg"
     DEPENDENCIES builtin_interfaces
   )

Update ``package.xml``:

.. code-block:: xml

   <build_depend>rosidl_default_generators</build_depend>
   <exec_depend>rosidl_default_runtime</exec_depend>
   <member_of_group>rosidl_interface_packages</member_of_group>
   <depend>builtin_interfaces</depend>

Step 3: Build Package
----------------------

.. code-block:: bash

   cd ~/ros2_ws
   colcon build --packages-select my_custom_msgs
   source install/setup.bash
   
   # Verify message exists
   ros2 interface show my_custom_msgs/msg/EstimateWithCovariance

Registering Converters with pykal
==================================

Basic Converter Pattern
-----------------------

.. code-block:: python

   from pykal.ros.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT
   from my_custom_msgs.msg import EstimateWithCovariance
   import numpy as np
   
   # ROS → NumPy converter
   def estimate_to_numpy(msg):
       """Convert EstimateWithCovariance to NumPy array."""
       state = np.array(msg.state)
       cov = np.array(msg.covariance).reshape(msg.state_dim, msg.state_dim)
       return np.concatenate([state, cov.flatten()])
   
   # NumPy → ROS converter
   def numpy_to_estimate(arr):
       """Convert NumPy array to EstimateWithCovariance."""
       msg = EstimateWithCovariance()
       
       # Assuming arr = [state..., covariance_flat...]
       # Need to know state dimension
       state_dim = 3  # Example: 3D state
       
       msg.state = arr[:state_dim].tolist()
       msg.covariance = arr[state_dim:].tolist()
       msg.state_dim = state_dim
       msg.stamp = ...  # Handle timestamp
       
       return msg
   
   # Register converters
   ROS2PY_DEFAULT[EstimateWithCovariance] = estimate_to_numpy
   PY2ROS_DEFAULT[EstimateWithCovariance] = numpy_to_estimate

**Key Points**:

- ROS → NumPy should return a 1D NumPy array
- NumPy → ROS should return a populated message object
- Handle timestamps if message includes them
- Convert lists to NumPy arrays and vice versa

Example: Multi-Sensor Fusion Message
-------------------------------------

**Message Definition**:

.. code-block:: text

   # msg/SensorFusion.msg
   geometry_msgs/Vector3 mocap      # Motion capture position
   std_msgs/Float64 barometer       # Barometer altitude
   geometry_msgs/Vector3 imu        # IMU velocities
   builtin_interfaces/Time stamp

**Converters**:

.. code-block:: python

   from my_custom_msgs.msg import SensorFusion
   from pykal.ros.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT
   import numpy as np
   
   def sensor_fusion_to_numpy(msg):
       """Convert SensorFusion to [mocap_x, mocap_y, mocap_z, baro, imu_x, imu_y, imu_z]."""
       return np.array([
           msg.mocap.x, msg.mocap.y, msg.mocap.z,
           msg.barometer.data,
           msg.imu.x, msg.imu.y, msg.imu.z
       ])
   
   def numpy_to_sensor_fusion(arr):
       """Convert NumPy [7] array to SensorFusion."""
       from geometry_msgs.msg import Vector3
       from std_msgs.msg import Float64
       from builtin_interfaces.msg import Time
       
       msg = SensorFusion()
       
       msg.mocap = Vector3(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
       msg.barometer = Float64(data=float(arr[3]))
       msg.imu = Vector3(x=float(arr[4]), y=float(arr[5]), z=float(arr[6]))
       msg.stamp = Time()  # Use current time or pass as parameter
       
       return msg
   
   ROS2PY_DEFAULT[SensorFusion] = sensor_fusion_to_numpy
   PY2ROS_DEFAULT[SensorFusion] = numpy_to_sensor_fusion

Example: Control Command Message
---------------------------------

**Message Definition**:

.. code-block:: text

   # msg/ControlCommand.msg
   float64[] forces       # Force commands [Fx, Fy, Fz]
   float64[] torques      # Torque commands [Tx, Ty, Tz]
   uint8 control_mode     # 0=position, 1=velocity, 2=force
   builtin_interfaces/Time stamp

**Converters**:

.. code-block:: python

   from my_custom_msgs.msg import ControlCommand
   
   def control_command_to_numpy(msg):
       """Convert to [Fx, Fy, Fz, Tx, Ty, Tz, mode]."""
       forces = np.array(msg.forces)
       torques = np.array(msg.torques)
       mode = np.array([msg.control_mode])
       return np.concatenate([forces, torques, mode])
   
   def numpy_to_control_command(arr):
       """Convert NumPy [7] array to ControlCommand."""
       msg = ControlCommand()
       msg.forces = arr[:3].tolist()
       msg.torques = arr[3:6].tolist()
       msg.control_mode = int(arr[6])
       msg.stamp = ...  # Handle timestamp
       return msg
   
   ROS2PY_DEFAULT[ControlCommand] = control_command_to_numpy
   PY2ROS_DEFAULT[ControlCommand] = numpy_to_control_command

Using Custom Messages with ROSNode
===================================

Basic Example
-------------

.. code-block:: python

   import rclpy
   from pykal.ros.ros_node import ROSNode
   from my_custom_msgs.msg import EstimateWithCovariance
   from pykal.ros.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT
   import numpy as np
   
   # 1. Register converters (do this once at module level)
   ROS2PY_DEFAULT[EstimateWithCovariance] = estimate_to_numpy
   PY2ROS_DEFAULT[EstimateWithCovariance] = numpy_to_estimate
   
   # 2. Create callback using custom message
   def kalman_filter_callback(tk, measurement):
       # measurement is NumPy array (converted automatically)
       # ... run Kalman filter
       
       # Return estimate (will be converted to EstimateWithCovariance)
       state = np.array([x, y, z])
       cov = P.flatten()
       return {'estimate': np.concatenate([state, cov])}
   
   # 3. Create ROSNode with custom message types
   kf_node = ROSNode(
       node_name='kalman_filter',
       callback=kalman_filter_callback,
       subscribes_to=[
           ('/measurement', EstimateWithCovariance, 'measurement')
       ],
       publishes_to=[
           ('estimate', EstimateWithCovariance, '/estimate')
       ],
       rate_hz=10.0
   )
   
   # 4. Run node
   rclpy.init()
   kf_node.create_node()
   kf_node.start()
   rclpy.spin(kf_node._node)

Complete Example: Sensor Fusion System
---------------------------------------

.. code-block:: python

   from my_custom_msgs.msg import SensorFusion
   from nav_msgs.msg import Odometry
   from pykal import DynamicalSystem
   from pykal.algorithm_library.estimators.kf import KF
   import numpy as np
   
   # Register converter (once)
   ROS2PY_DEFAULT[SensorFusion] = sensor_fusion_to_numpy
   PY2ROS_DEFAULT[SensorFusion] = numpy_to_sensor_fusion
   
   # Create Kalman filter callback
   def fusion_callback(tk, sensors):
       # sensors is [mocap_x, mocap_y, mocap_z, baro, imu_x, imu_y, imu_z]
       
       # Extract sensor data
       mocap = sensors[:3]
       baro = sensors[3]
       imu = sensors[4:]
       
       # Fuse sensors with KF
       # ... (KF implementation)
       
       # Return estimate as Odometry
       estimate = np.concatenate([
           position, orientation, linear_vel, angular_vel
       ])
       return {'estimate': estimate}
   
   # Create node
   fusion_node = ROSNode(
       node_name='sensor_fusion',
       callback=fusion_callback,
       subscribes_to=[
           ('/sensors', SensorFusion, 'sensors')
       ],
       publishes_to=[
           ('estimate', Odometry, '/estimate')
       ],
       rate_hz=100.0
   )

Advanced Topics
===============

Handling Timestamps
-------------------

For messages with timestamps, use ROS2 time utilities:

.. code-block:: python

   from builtin_interfaces.msg import Time
   import rclpy.time
   
   def numpy_to_custom_msg(arr, node=None):
       msg = CustomMsg()
       msg.data = arr.tolist()
       
       # Get current ROS time
       if node is not None:
           now = node.get_clock().now()
           msg.stamp = now.to_msg()
       else:
           msg.stamp = Time()  # Zero timestamp
       
       return msg

This requires passing the node to the converter. Currently, ``pykal`` doesn't support this directly.

**Workaround**: Set timestamp to zero or use wall-clock time:

.. code-block:: python

   import time
   
   def numpy_to_custom_msg(arr):
       msg = CustomMsg()
       msg.data = arr.tolist()
       
       # Use wall-clock time (not recommended for simulation!)
       t = time.time()
       msg.stamp.sec = int(t)
       msg.stamp.nanosec = int((t % 1) * 1e9)
       
       return msg

Variable-Length Arrays
----------------------

For messages with variable-length arrays:

.. code-block:: python

   # Message: float64[] data
   
   def msg_to_numpy(msg):
       return np.array(msg.data)  # Simple!
   
   def numpy_to_msg(arr):
       msg = CustomMsg()
       msg.data = arr.flatten().tolist()  # Flatten first
       return msg

Nested Messages
---------------

For messages containing other messages:

.. code-block:: python

   # Message:
   #   geometry_msgs/Pose pose
   #   geometry_msgs/Twist velocity
   
   def nested_to_numpy(msg):
       # Flatten nested structure
       pose = np.array([
           msg.pose.position.x,
           msg.pose.position.y,
           msg.pose.position.z,
           msg.pose.orientation.x,
           msg.pose.orientation.y,
           msg.pose.orientation.z,
           msg.pose.orientation.w
       ])
       
       twist = np.array([
           msg.velocity.linear.x,
           msg.velocity.linear.y,
           msg.velocity.linear.z,
           msg.velocity.angular.x,
           msg.velocity.angular.y,
           msg.velocity.angular.z
       ])
       
       return np.concatenate([pose, twist])

Array Messages
--------------

For arrays of messages:

.. code-block:: python

   # Message: CustomMsg[] array
   
   def array_to_numpy(msg):
       # Stack all elements
       return np.vstack([
           np.array([elem.x, elem.y, elem.z])
           for elem in msg.array
       ]).flatten()
   
   def numpy_to_array(arr):
       msg = ArrayMsg()
       arr_2d = arr.reshape(-1, 3)  # Reshape to (N, 3)
       
       msg.array = [
           CustomMsg(x=row[0], y=row[1], z=row[2])
           for row in arr_2d
       ]
       return msg

Best Practices
==============

1. **Flatten to 1D Arrays**

   Always return 1D NumPy arrays from ROS → NumPy converters:

   .. code-block:: python

      # GOOD
      return np.concatenate([position, orientation.flatten()])
      
      # BAD
      return np.vstack([position, orientation])  # 2D!

2. **Use** ``.tolist()`` **for ROS Messages**

   ROS expects Python lists, not NumPy arrays:

   .. code-block:: python

      # GOOD
      msg.data = arr.tolist()
      
      # BAD (may work but not guaranteed)
      msg.data = arr

3. **Document Expected Array Format**

   .. code-block:: python

      def custom_to_numpy(msg):
          """
          Convert CustomMsg to NumPy array.
          
          Returns
          -------
          np.ndarray, shape (10,)
              [x, y, z, qx, qy, qz, qw, vx, vy, vz]
          """
          # ...

4. **Register Early**

   Register converters at module import time, not inside functions:

   .. code-block:: python

      # my_package/converters.py
      from pykal.ros.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT
      from my_custom_msgs.msg import CustomMsg
      
      ROS2PY_DEFAULT[CustomMsg] = custom_to_numpy
      PY2ROS_DEFAULT[CustomMsg] = numpy_to_custom
      
      # Then in main script:
      import my_package.converters  # Registers automatically

5. **Test Converters Independently**

   .. code-block:: python

      # Test round-trip conversion
      msg_original = CustomMsg(...)
      arr = custom_to_numpy(msg_original)
      msg_reconstructed = numpy_to_custom(arr)
      
      # Verify equality
      assert msg_original == msg_reconstructed

Common Pitfalls
===============

**Pitfall 1: Wrong Array Shape**

.. code-block:: python

   # WRONG: Returning 2D array
   def bad_converter(msg):
       return np.array([[msg.x, msg.y, msg.z]])  # Shape (1, 3)
   
   # CORRECT: Return 1D
   def good_converter(msg):
       return np.array([msg.x, msg.y, msg.z])  # Shape (3,)

**Pitfall 2: Forgetting to Register**

.. code-block:: python

   # Converter defined but not registered!
   def my_converter(msg):
       return np.array([...])
   
   # Must add:
   ROS2PY_DEFAULT[MyMsg] = my_converter

**Pitfall 3: Type Mismatches**

.. code-block:: python

   # Message expects float64[]
   msg.data = [1, 2, 3]  # Python ints, may fail!
   
   # Better:
   msg.data = [float(x) for x in arr]
   # Or:
   msg.data = arr.astype(float).tolist()

**Pitfall 4: Modifying Converters After Node Creation**

.. code-block:: python

   # BAD: Node already created with old converter
   node.create_node()
   ROS2PY_DEFAULT[MyMsg] = new_converter  # Too late!
   
   # GOOD: Register before node creation
   ROS2PY_DEFAULT[MyMsg] = new_converter
   node.create_node()

Example: Complete Custom Message Workflow
==========================================

**1. Create message package**:

.. code-block:: bash

   ros2 pkg create --build-type ament_cmake robot_msgs
   cd robot_msgs/msg
   echo "float64[] state
   float64[] covariance
   uint32 dim" > StateEstimate.msg
   
   cd ..
   # Edit CMakeLists.txt and package.xml
   cd ~/ros2_ws && colcon build

**2. Create converter module**:

.. code-block:: python

   # robot_utils/converters.py
   from pykal.ros.ros2py_py2ros import ROS2PY_DEFAULT, PY2ROS_DEFAULT
   from robot_msgs.msg import StateEstimate
   import numpy as np
   
   def state_estimate_to_numpy(msg):
       """[state..., covariance_flat...]"""
       state = np.array(msg.state)
       cov = np.array(msg.covariance)
       return np.concatenate([state, cov])
   
   def numpy_to_state_estimate(arr):
       """Assumes dim=3 for simplicity."""
       msg = StateEstimate()
       msg.dim = 3
       msg.state = arr[:3].tolist()
       msg.covariance = arr[3:].tolist()
       return msg
   
   # Auto-register on import
   ROS2PY_DEFAULT[StateEstimate] = state_estimate_to_numpy
   PY2ROS_DEFAULT[StateEstimate] = numpy_to_state_estimate

**3. Use in ROSNode**:

.. code-block:: python

   # main.py
   import robot_utils.converters  # Registers converters
   from robot_msgs.msg import StateEstimate
   from pykal.ros.ros_node import ROSNode
   import numpy as np
   
   def estimator_callback(tk, measurement):
       # measurement auto-converted to NumPy
       # ... run filter
       state = np.array([x, y, z])
       cov = P.flatten()
       return {'estimate': np.concatenate([state, cov])}
   
   node = ROSNode(
       node_name='estimator',
       callback=estimator_callback,
       subscribes_to=[('/measurement', StateEstimate, 'measurement')],
       publishes_to=[('estimate', StateEstimate, '/estimate')],
       rate_hz=10.0
   )
   
   import rclpy
   rclpy.init()
   node.create_node()
   node.start()
   rclpy.spin(node._node)

Summary
=======

**Custom Message Workflow**:

1. ✓ Define ``.msg`` file in ROS2 package
2. ✓ Build package with ``colcon build``
3. ✓ Write ROS → NumPy converter (returns 1D array)
4. ✓ Write NumPy → ROS converter (returns message object)
5. ✓ Register both converters in registries
6. ✓ Use message type in ROSNode ``subscribes_to``/``publishes_to``
7. ✓ Test converters independently before integration

**Key Reminders**:

- ROS → NumPy must return **1D NumPy array**
- NumPy → ROS must return **populated message object**
- Register converters **before** creating nodes
- Use ``.tolist()`` when assigning NumPy arrays to message fields
- Document expected array format in docstrings
- Test round-trip conversion

**See Also**:

- :doc:`ROSNode Documentation <modules>`
- :doc:`Troubleshooting Guide <troubleshooting>`
- ROS2 Interface Tutorial: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html

:doc:`← Python to ROS <index>`
