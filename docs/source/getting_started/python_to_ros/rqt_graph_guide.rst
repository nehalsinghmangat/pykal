:doc:`← Python to ROS <index>`

============================
Visualizing ROS Graphs with rqt_graph
============================

Throughout the ROS deployment tutorials, we've emphasized that **the ROS graph mirrors the block diagram**. The ``rqt_graph`` tool makes this connection visual and helps debug ROS systems.

What is rqt_graph?
==================

``rqt_graph`` is a ROS tool that displays:

- **Nodes**: Circles representing running ROS nodes
- **Topics**: Arrows showing topic connections
- **Message flow**: Direction of data flow
- **Node groupings**: Namespaces and node hierarchies

Installation
============

``rqt_graph`` comes with ROS2 desktop installation:

.. code-block:: bash

   # For Ubuntu with ROS2
   sudo apt install ros-humble-rqt-graph
   
   # Or if using different ROS2 distribution
   sudo apt install ros-<distro>-rqt-graph

Usage
=====

Basic Usage
-----------

1. **Start your ROS system** (all nodes running)
2. **Open rqt_graph** in a new terminal:

.. code-block:: bash

   rqt_graph

3. **Configure the view**:
   - Top dropdown: Select "Nodes/Topics (all)"
   - Check/uncheck boxes to show/hide elements
   - Click "Refresh" to update the graph

Visualizing the TurtleBot Software System
------------------------------------------

For the TurtleBot ROS deployment (software-only), you should see:

::

    ┌─────────────────┐
    │ waypoint_       │
    │ generator       │──→ /reference
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │ velocity_       │
    │ controller      │──→ /cmd_vel
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │ turtlebot_      │──→ /odom
    │ simulator       │──→ /true_state
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │ kalman_         │──→ /estimate
    │ filter          │
    └─────────────────┘
           ↑
           └─────(feedback)

**What to notice**:

- **4 nodes**: waypoint_generator, velocity_controller, turtlebot_simulator, kalman_filter
- **5 topics**: /reference, /cmd_vel, /odom, /true_state, /estimate
- **Feedback loop**: /estimate → velocity_controller → /cmd_vel

Visualizing TurtleBot with Gazebo
----------------------------------

After integrating with Gazebo, the graph changes:

::

    ┌─────────────────┐
    │ waypoint_       │
    │ generator       │──→ /reference
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │ velocity_       │
    │ controller      │──→ /cmd_vel
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │ GAZEBO          │──→ /odom
    │ (physics +      │──→ /scan
    │  sensors)       │──→ /imu
    └─────────────────┘──→ /camera
           ↓
    ┌─────────────────┐
    │ kalman_         │──→ /estimate
    │ filter          │
    └─────────────────┘
           ↑
           └─────(feedback)

**Key differences**:

- ❌ **Removed**: turtlebot_simulator node
- ✓ **Added**: Gazebo node (with many topics!)
- ✓ **Unchanged**: waypoint_generator, velocity_controller, kalman_filter

This visual confirms: **same control nodes, different sensor source!**

Visualizing Crazyflie Multi-Sensor Fusion
------------------------------------------

For the Crazyflie software simulation with 3 sensors:

::

    ┌─────────────┐
    │ setpoint_   │
    │ generator   │──→ /setpoint
    └─────────────┘
           ↓
    ┌─────────────┐
    │ position_   │
    │ controller  │──→ /cmd_vel
    └─────────────┘
           ↓
    ┌─────────────┐
    │ crazyflie_  │──→ /mocap ──┐
    │ simulator   │──→ /baro  ──┼─→ ┌─────────────┐
    │             │──→ /imu   ──┘   │ kalman_     │──→ /estimate
    └─────────────┘                 │ filter      │
                                     └─────────────┘
                                            ↑
                                            └──(feedback)

**What to notice**:

- **3 sensor topics** converge at kalman_filter
- Demonstrates **multi-sensor fusion pattern**
- All 3 arrows point to one node (fusion!)

Useful rqt_graph Features
==========================

Filtering Nodes
---------------

Use the dropdowns to focus on specific parts:

- **Nodes only**: Hide topics, show just node connectivity
- **Nodes/Topics (active)**: Only show active connections
- **Nodes/Topics (all)**: Show everything, including unused topics

Hiding Debug Topics
-------------------

ROS creates many debug topics. To hide them:

1. Click the "Hide" checkbox
2. Add patterns like: ``/rosout``, ``/parameter_events``, ``/tf``

This cleans up the graph to show only your application topics.

Refreshing the Graph
--------------------

The graph doesn't auto-update. After starting/stopping nodes:

- Click the **Refresh** button (circular arrow)
- Or press ``Ctrl+R``

Layout Options
--------------

If nodes overlap:

- Try different layout algorithms (dropdown menu)
- Manually drag nodes to reorganize
- Zoom in/out with mouse wheel

Comparing Architectures
========================

Software vs Gazebo vs Hardware
-------------------------------

Use ``rqt_graph`` to **compare** before/after Gazebo integration:

1. **Take screenshot** of software simulation graph
2. **Stop software nodes**, start Gazebo
3. **Take screenshot** of Gazebo integration graph
4. **Compare side-by-side**:

   - What nodes were removed?
   - What nodes were added?
   - Which nodes stayed the same?

Example comparison table:

+-------------------+------------------------+--------------------+
| Component         | Software Simulation    | Gazebo Integration |
+===================+========================+====================+
| **Simulator**     | turtlebot_simulator    | Gazebo             |
+-------------------+------------------------+--------------------+
| **Controller**    | velocity_controller    | velocity_controller|
+-------------------+------------------------+--------------------+
| **Observer**      | kalman_filter          | kalman_filter      |
+-------------------+------------------------+--------------------+
| **Topics**        | /odom, /true_state     | /odom, /scan, /imu |
+-------------------+------------------------+--------------------+

Debugging with rqt_graph
=========================

Common Issues
-------------

**Problem**: "My node doesn't receive messages"

**Solution**: Check rqt_graph:

1. Is the publisher node visible?
2. Is the topic arrow connected?
3. Is the topic name spelled correctly?
4. Are there multiple publishers? (conflict)

**Problem**: "Topics exist but no data flows"

**Solution**: rqt_graph shows **potential** connections, not actual data flow. Check:

1. Topic types match (``ros2 topic info /topic_name``)
2. QoS settings compatible
3. Messages actually being published (``ros2 topic echo``)

**Problem**: "Gazebo node not appearing"

**Solution**:

1. Wait 5-10 seconds for Gazebo to initialize
2. Click "Refresh" in rqt_graph
3. Check "Nodes/Topics (all)" to show hidden nodes

Advanced: Recording Graph Changes
==================================

To document architecture evolution:

.. code-block:: bash

   # Start with software simulation
   # Take screenshot: Ctrl+Shift+PrtScn
   
   # Stop software, start Gazebo
   # Refresh rqt_graph
   # Take screenshot again
   
   # Compare images side-by-side

Or automate with:

.. code-block:: bash

   # Save graph as image
   rqt_graph --perspective-file my_graph.perspective
   
   # Export as DOT file
   ros2 run rqt_graph rqt_graph --dot-file graph.dot

Alternative Tools
=================

While ``rqt_graph`` is the standard, alternatives exist:

- **rqt_tf_tree**: Visualize coordinate transform tree
- **plotjuggler**: Time-series data visualization
- **rviz2**: 3D robot state visualization
- **ros2 topic list**: Command-line topic inspection

For **understanding system architecture**, ``rqt_graph`` is best.

Summary
=======

``rqt_graph`` is essential for:

✓ **Verifying** your ROS graph matches the block diagram
✓ **Debugging** connection issues
✓ **Documenting** system architecture
✓ **Comparing** software vs Gazebo vs hardware
✓ **Understanding** multi-sensor fusion patterns

**Best Practice**: Always run ``rqt_graph`` when:

- Starting a new ROS system
- Debugging connection issues
- Integrating with Gazebo
- Deploying to hardware
- Teaching/presenting ROS systems

The visual feedback confirms that **theory (block diagrams) = practice (ROS graphs)**!

:doc:`← Python to ROS <index>`
