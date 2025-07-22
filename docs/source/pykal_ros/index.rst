=========
pykal_ros
=========

``pykal_ros``  is the bridge between software and simulation (using the ROS API). It is a Python package that allows us to wrap the control theoretic objects defined in ``pykal_core`` into standard ROS nodes. 


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
   
Signal Nodes
------------

.. graphviz::
   :align: center

   digraph SignalNodes {
       rankdir=LR;
       node [shape=box, style=filled, fontname="Helvetica"];
       graph [splines=ortho, nodesep=1.0, ranksep=1.0];

       // --- Python Group ---
       subgraph cluster_python {
           label = "Python (Offline)";
           style = filled;
           color = green;
           fillcolor = lightgreen;

           signal_class [label="Signal (class)"];
           signal_object [label="signal (object)"];
           signal_dill [label="signal.dill"];

           signal_class -> signal_object [label="instantiates"];
           signal_object -> signal_dill [label="saved to"];
       }

       // --- ROS Group ---
       subgraph cluster_ros {
           label = "ROS 2 Runtime";
           style = filled;
           color = blue;
           fillcolor = lightblue;

           generator_py [label="generate_signal_node.py"];
           signal_node [label="signal_node (rclpy)"];
           input_topic [label="Topic"];
           output_topic [label="Topic"];

           generator_py -> signal_node [label="spawns"];
           signal_node -> input_topic [label="subscribes"];
           signal_node -> output_topic [label="publishes"];
       }

       // --- Cross-link with elbow and label ---
       signal_dill -> generator_py [
           style=dotted,
           arrowhead=normal,
           tailport=s,
           headport=n,
           constraint=false,
       ];
   }

       
Observer Nodes
--------------
.. graphviz::
   :align: center

   digraph ObserverNodes {
       rankdir=LR;
       node [shape=box, style=filled, fontname="Helvetica"];
       graph [splines=ortho, nodesep=1.0, ranksep=1.0];

       // --- Python Group ---
       subgraph cluster_python {
           label = "Python (Offline)";
           style = filled;
           color = green;
           fillcolor = lightgreen;

           observer_class [label="Observer (class)"];
           observer_object [label="observer (object)"];
           observer_dill [label="observer.dill"];

           observer_class -> observer_object [label="instantiates"];
           observer_object -> observer_dill [label="saved to"];
       }

       // --- ROS Group ---
       subgraph cluster_ros {
           label = "ROS 2 Runtime";
           style = filled;
           color = blue;
           fillcolor = lightblue;

           generator_py [label="generate_observer_node.py"];
           observer_node [label="observer_node (rclpy)"];
           input_topic [label="Topic"];
           output_topic [label="Topic"];

           generator_py -> observer_node [label="spawns"];
           observer_node -> input_topic [label="subscribes"];
           observer_node -> output_topic [label="publishes"];
       }

       // --- Cross-link with elbow and label ---
       observer_dill -> generator_py [
           style=dotted,
           arrowhead=normal,
           tailport=s,
           headport=n,
           constraint=false,
       ];
   }


Controller Nodes
----------------
.. graphviz::
   :align: center

   digraph ControllerNodes {
       rankdir=LR;
       node [shape=box, style=filled, fontname="Helvetica"];
       graph [splines=ortho, nodesep=1.0, ranksep=1.0];

       // --- Python Group ---
       subgraph cluster_python {
           label = "Python (Offline)";
           style = filled;
           color = green;
           fillcolor = lightgreen;

           controller_class [label="Controller (class)"];
           controller_object [label="controller (object)"];
           controller_dill [label="controller.dill"];

           controller_class -> controller_object [label="instantiates"];
           controller_object -> controller_dill [label="saved to"];
       }

       // --- ROS Group ---
       subgraph cluster_ros {
           label = "ROS 2 Runtime";
           style = filled;
           color = blue;
           fillcolor = lightblue;

           generator_py [label="generate_controller_node.py"];
           controller_node [label="controller_node (rclpy)"];
           input_topic [label="Topic"];
           output_topic [label="Topic"];

           generator_py -> controller_node [label="spawns"];
           controller_node -> input_topic [label="subscribes"];
           controller_node -> output_topic [label="publishes"];
       }

       // --- Cross-link with elbow and label ---
       controller_dill -> generator_py [
           style=dotted,
           arrowhead=normal,
           tailport=s,
           headport=n,
           constraint=false,
       ];
   }

