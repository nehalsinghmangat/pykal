pykal
=====

.. epigraph::

   "In theory, there is no difference between theory and practice. In practice, there is."

   -- Yogi Berra

``pykal`` is a Python framework designed to bridge the gap between the theoretical formulations of control systems and their respective implementations in robotics hardware. Roughly speaking, there are four steps to building a robot:

.. graphviz::
   :align: center

   digraph RobotFlow {
       rankdir=LR;
       node [shape=box, style=filled, fillcolor=white, fontname="Helvetica"];

       Theory    [label="Theory"];
       Software  [label="Software"];
       Simulation[label="Simulation"];
       Hardware  [label="Hardware"];

       edge [color=red, penwidth=2,style=dotted];

       Theory    -> Software;
       Software  -> Simulation;
       Simulation-> Hardware;
   }

But progress is never linear, and our workflow will inevitably look something like this:

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

       // Feedback arrows (dotted red)
       edge [color=red, style=dotted, penwidth=1];
       Hardware   -> Simulation;
       Simulation -> Software;
       Software   -> Theory;
   }


``pykal`` aims to make cycling through this workflow as painless as possible. Designed for students, professionals, and academics alike, this framework won't cure cancer, but it can do the next best thing: make building robots easier. 

If you are new to this package and want an idea of what ``pykal`` can do, please click here: :doc:`quickstart`.

If you need a refresher, click here: Example Notebook

For a three minute video, click here: Link to my presentation





