=====================
 The pykal framework
=====================

Recall the four stages of building a robot:


For a basic foundation in how to use the framework, start with the quickstart tutorial in pykal_core and then the quickstart tutorial in pykal_ros.

pykal_core
----------
**pykal_core** is the bridge between theory and software. To see the **pykal_core** documentation, including installation and tutorials, please click here: :doc:`pykal_core <../../pykal_core/index>`

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: space-between; gap: 2em; margin-top: 1em; margin-bottom: 1em;">
     <img src="./_static/system_input_observer_block_diagram.svg" alt="System Observer Block Diagram" style="width: 70%;">
     <img src="./_static/pendulum_torque_laser.svg" alt="Pendulum with Laser Sensor" style="width: 30%;">
   </div>

   
A Python package that enables intuitive and flexible modeling of control systems, **pykal_core** includes:

- standard control system tools (e.g.signal generators, Kalman Filters, PID Controllers)
- support for arbitrary extension and modification of standard algorithms into cutting-edge research variants (e.g. Observability-Informed Partial-Update Extended-Kalman Filter)
- support for arbitrary function models (e.g. analytical, ML-based) with a well-defined functional interface

pykal_ros
---------

**pykal_ros**  is the bridge between software and simulation (using the ROS API). To see the **pykal_ros** documentation, including installation and tutorials, please click here: :doc:`pykal_ros <../../pykal_ros/index>`   

.. graphviz::
   :align: center

   digraph RobotLoop {
       rankdir=LR;
       node [shape=box, style=filled, fillcolor=white, fontname="Helvetica"];

       // pykal supercluster
       subgraph cluster_pykal_ros {
       label = "pykal_ros";
       fontcolor = "#a80000";
       style = dotted;
       fillcolor = none;

       Software    [label="Software"];
       Simulation  [label="Simulation"];
       }

       // Forward arrows (solid black)
       edge [color=red,style=dotted, penwidth=1];
       Software     -> Simulation;

       // Feedback arrows (dotted red)
       edge [color=red, style=dotted, penwidth=1];
       Simulation   -> Software;
   }
   
A Python package that wraps the control theoretic objects defined in **pykal_core** into standard ROS nodes,  **pykal_ros** includes:

- ``generate_[object]_node.py`` files that, when called with  ``ros2 run --rosargs``, spin up ros nodes that encapsulate user-defined ``pykal_core`` objects.
- ``generate_meta_[object]_node.py`` files that, when called with  ``ros2 run --rosargs``, control the publishing of [object]-nodes

   For example, one can define two ``observer_nodes`` for a system, one of which uses the standard EKF and the other a more computationally expensive KF-variant for highly nonlinear trajectories.

   An ``observer_meta_node`` may monitor the nonlinearities of the system through some user-defined metric; once the nonlinearity exceeds a threshold, the ``observer_meta_node`` can halt the computations and publications occuring in the standard EKF node and start the same in the expensive KF-variant node. Once the nonlinearity has passed, the ``observer_meta_node`` can than switch the nodes back to their previous functioning.



Gazebo
------
**Gazebo** is a third-party open-source robotics simulator which is compatible with ROS. It is the final bridge between simulation and hardware. To learn how to use the **Gazebo**  and how to interface it with ROS, please consult the official **Gazebo** website: `https://gazebosim.org/ <https://gazebosim.org/>`_



.. graphviz::
   :align: center

   digraph RobotLoop {
       rankdir=LR;
       node [shape=box, style=filled, fillcolor=white, fontname="Helvetica"];

       // pykal supercluster
       subgraph cluster_gazebo {
       label = "Gazebo";
       fontcolor = "#a80000";
       style = dotted;
       fillcolor = none;

       Simulation  [label="Simulation"];
       Hardware  [label="Hardware"];       
       }

       // Forward arrows (solid black)
       edge [color=red,style=dotted, penwidth=1];
       Simulation  -> Hardware;

       // Feedback arrows (dotted red)
       edge [color=red, style=dotted, penwidth=1];
       Hardware   -> Simulation;
   }
   
----

:doc:`← On Building Robots <on_building_robots>` | :doc:`Conceptual Reference Index → <../index>`

----
