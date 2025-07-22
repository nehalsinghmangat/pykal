pykal: From Theory to Python to ROS
===================================

.. epigraph::

   "Cross a river once, swim; cross a river twice, build a boat; cross a river thrice, build a bridge.
   "
   
   -- old Punjabi saying

   
**pykal** is a Python framework designed to bridge the gap between the theoretical formulations of control systems and their respective implementation in robotics hardware. Roughly speaking, this is the road from brain to robot:
   
.. graphviz::
   :align: center

   digraph RobotLoop {
       rankdir=LR;
       node [shape=box, style=filled, fillcolor=white, fontname="Helvetica"];

       Theory    [label="Theory"];
       Software  [label="Software"];
       Simulation[label="Simulation"];
       Hardware  [label="Hardware"];

       // Forward arrows (solid black)
       edge [color=red,style=dotted, penwidth=1];
       Theory     -> Software;
       Software   -> Simulation;
       Simulation -> Hardware;
   }

.. raw:: html

   <div style="margin-top: 1.5em;"></div>
   
**pykal** aims to make every step forward, every step back, and every detour on this road as painless as possible. Designed for hobbyists, students, and academics alike, this framework won't cure cancer, but it can do the next best thing: make building robots easier.

To get started with **pykal**, or to explore the framework and its capabilities, click here: :doc:`Quickstart <quickstart_index>`.

For a 3-min video describing the motivation behind this framework and what it hopes to achieve, click here: video.

.. raw:: html

   <div style="display: flex; justify-content: space-around; align-items: flex-start; gap: 20px; margin-top: 1em; margin-bottom: 1em;">
     <figure style="width: 30%; text-align: center;">
       <img src="_static/turtlesim_software.gif" alt="Turtle Software" style="width: 100%;">
       <figcaption>Software</figcaption>
     </figure>
     <figure style="width: 30%; text-align: center;">
       <img src="_static/turtlesim_simulation.gif" alt="Turtle Simulation" style="width: 100%;">
       <figcaption>Simulation</figcaption>
     </figure>
     <figure style="width: 30%; text-align: center;">
       <img src="_static/turtlesim_hardware.gif" alt="Turtle Hardware" style="width: 100%;">
       <figcaption>Hardware</figcaption>
     </figure>
   </div>

.. raw:: html

   <div style="text-align: center; font-style: italic; margin-top: -0.2em; margin-bottom: 1.5em;">
     Figure 1: (left) The trajectory of a Turtlebot given certain control inputs in Python; (center) the trajectory of a Turtlebot given the same control inputs in the ROS Turtlesim simulator; (right) the trajectory of a real Turtlebot given the same ROS control inputs (trajectory traced by mocab).
   </div>   














