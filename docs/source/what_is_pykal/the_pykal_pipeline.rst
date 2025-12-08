==============================
 Overview: The pykal pipeline
==============================

Building a robot occurs in four steps, and ``pykal`` strives to make every step forward and every step back as painless as possible:

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

**Theory**. Before we can control anything more complicated than a paperclip, we need to cast our robot as a control system.

**Software**. Once we have a control system, we must model it effectively in software. This enables us to run simulations and fix potential shortcomings of our model.

**Simulation**. The world of simulation is beautiful and forgiving. When simulations fail, it is an inconvenience

**Hardware**. The world of hardware is neither beautiful nor forgiving. When hardware fails, it is a catastrophe.

Each of these domains are difficult in their own right, but it is on the bridges between them that good men die. This is what ``pykal`` aims to fix.
   
----

:doc:`← Robotics and Complexity <robotics_and_complexity>` | :doc:`Theory to Software → <control_algorithms_as_dynamical_systems>`

----
