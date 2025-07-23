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

   


pykal_ros
---------

**pykal_ros**  is the bridge between software and simulation (using the ROS API). To see the **pykal_ros** documentation, including installation and tutorials, please click here: 

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
