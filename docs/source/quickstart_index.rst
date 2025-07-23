Quickstart
==========

We recommend reading each section below to get an idea of the overall structure of the framework. Each section links to its respective package within the framework. After working through the tutorials in pykal_core and pykal_ros, you will be able to start designing your own control systems and implementing them on hardware.

For a motivating background on the origin of the **pykal** framework, please click here

Overview of pykal
-----------------
**pykal** operates on the assumption that this is the road from brain to robot:
   
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
   
**pykal** aims to make every step forward, every step back, and every detour on this road as painless as possible. To be clear, **pykal** is not a package in itself; rather, **pykal** refers to the framework which is composed of two packages: pykal_core and pykal_ros. Each package is described below. 


pykal_core
^^^^^^^^^^

``pykal_core`` is the bridge between theory and software. It is a Python package that enables intuitive and flexible modeling of control systems. For full documentation, including installation and tutorials, click here: :doc:`pykal_core <./pykal_core/index.rst>`.   

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
   
**pykal_core** includes:

- standard control system tools (e.g.signal generators, Kalman Filters, PID Controllers)
- support for arbitrary extension and modification of standard algorithms into cutting-edge research variants (e.g. Observability-Informed Partial-Update Extended-Kalman Filter)
- support for arbitrary function models (e.g. analytical, ML-based) with a well-defined functional interface

  
pykal_ros
^^^^^^^^^

``pykal_ros``  is the bridge between software and simulation (using the ROS API). It is a Python package that allows us to wrap the control theoretic objects defined in ``pykal_core`` into standard ROS nodes. For full documentation, including installation and tutorials, click here: :doc:`pykal_ros <./pykal_ros/index.rst>`.   


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


   
**pykal_ros** includes:

- ``generate_[object]_node.py`` files that, when called with  ``ros2 run --rosargs``, spin up ros nodes that encapsulate user-defined ``pykal_core`` objects.
- ``generate_meta_[object]_node.py`` files that, when called with  ``ros2 run --rosargs``, control the publishing of [object]-nodes

   For example, one can define two ``observer_nodes`` for a system, one of which uses the standard EKF and the other a more computationally expensive KF-variant for highly nonlinear trajectories.

   An ``observer_meta_node`` may monitor the nonlinearities of the system through some user-defined metric; once the nonlinearity exceeds a threshold, the ``observer_meta_node`` can halt the computations and publications occuring in the standard EKF node and start the same in the expensive KF-variant node. Once the nonlinearity has passed, the ``observer_meta_node`` can than switch the nodes back to their previous functioning.





