Framework Overview
------------------
To be clear, ``pykal`` is not a Python package; rather, it is a framework which is composed of **two** packages:  :doc:`pykal_core <./pykal_core/index>` and :doc:`pykal_ros <./pykal_ros/index>`.

``pykal`` assumes this is the road from brain to robot:
   
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
   
``pykal`` aims to make every step forward, every step back, and every detour on this road as painless as possible.

pykal_core
^^^^^^^^^^

``pykal_core`` is the bridge between theory and software. It is a Python package that enables intuitive and flexible modeling of control systems. For full documentation, including installation and tutorials, click here: :doc:`pykal_core <./pykal_core/index>`.   

.. graphviz::
   :align: center

   digraph RobotLoop {
       rankdir=LR;
       node [shape=box, style=filled, fillcolor=white, fontname="Helvetica"];

       // pykal supercluster
       subgraph cluster_pykal {
       label = "pykal";
       fontcolor = "#a80000";
       style = dotted;
       fillcolor = none;

       Theory    [label="Theory"];
       Software  [label="Software"];
       }

       // Forward arrows (solid black)
       edge [color=red,style=dotted, penwidth=1];
       Theory     -> Software;

       // Feedback arrows (dotted red)
       edge [color=red, style=dotted, penwidth=1];
       Software   -> Theory;
   }
   
  
pykal_ros
^^^^^^^^^

``pykal_ros``  is the bridge between software and simulation (using the ROS API). It is a Python package that allows us to wrap the control theoretic objects defined in ``pykal_core`` into standard ROS nodes. For full documentation, including installation and tutorials, click here: :doc:`pykal_ros <./pykal_ros/index>`.   


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


I often find that knowing the motivation and history behind a particular package/framework useful for understanding why it is designed the way it is. Towards that end, you may find reading the next section useful.


Otherwise, please follow the links to either ``pykal`` package above.

----

:doc:`← On Building Robots <on_building_robots>` | :doc:`Why did you make pykal? → <why_did_you_make_pykal>`

----   
