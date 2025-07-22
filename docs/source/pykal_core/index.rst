pykal_core
==========
``pykal_core`` is the bridge between theory and software. It is a Python package that enables intuitive and flexible modeling of control systems.

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

..  toctree::
   :maxdepth: 2
   :caption: User Guide

   installation

..  toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart
   
.. toctree::
   :maxdepth: 1
   :caption: API

   apiref/index

.. toctree::
   :maxdepth: 1
   :caption: Conceptual Reference

   conceptref/dynamical_systems/index
   conceptref/state_estimation/index
   conceptref/kalman_filters/index
   









   
