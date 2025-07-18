Class Inheritance and Module Usage
====================================
The following diagrams show the inheritance and module relationships within the base pykal framework. Beneath each diagram is a table of links explaining the design and rationale of important classes, how each class interfaces with other classes, and so on. 

Base pykal
^^^^^^^^^^^

.. graphviz::
   :caption: Class Inheritance and Module Usage

   digraph ClassInheritance {
     node [shape=box, fontname="Helvetica", style=filled, fillcolor=lightgray];
     rankdir=TB;
     splines=ortho;

     // pykal/ supercluster
     subgraph cluster_pykal {
       label = "pykal/";
       fontcolor = "#a80000";
       style = dotted;
       fillcolor = none;

   // control_system/
       subgraph cluster_module_control_system {
          label = "control_system/";
      fontcolor = "#a80000";
       color = "dimgray";
       fillcolor = none;
       // system.py
       subgraph cluster_system {
         label = "system.py";
         style = rounded;
         fontcolor = "#a80000";
         System [label="System", shape=note, fillcolor=lightblue];
       }

       // observer.py
       subgraph cluster_observer {
         label = "observer.py";
         style = rounded;
         fontcolor = "#a80000";
         Observer [label="Observer", shape=note, fillcolor=lightblue];
       }

       // controller.py
       subgraph cluster_controller {
         label = "controller.py";
         style = rounded;
         fontcolor = "#a80000";
         Controller [label="Controller", shape=note, fillcolor=lightblue];
       }       
     }
     // pykal/ supercluster
     subgraph cluster_pykal {
       label = "utils/";
       fontcolor = "#a80000";
       color = "dimgray";
       fillcolor = none;
       // compute.py
       subgraph cluster_compute {
         label = "compute.py";
         style = rounded;
         fontcolor = "#a80000";
	 Observability [label="Observability", shape=note, fillcolor=white];
	 Controllability [label="Controllability", shape=note, fillcolor=white];
         Error [label="Error", shape=note, fillcolor=white];
       }}

     subgraph cluster_module_control {
          label = "control/";
      fontcolor = "#a80000";
       color = "dimgray";
       fillcolor = none;

       subgraph cluster_module_est {
          label = "pid.py";
      fontcolor = "#a80000";
       color = "dimgray";
       fillcolor = none;
      PID [label="PID", shape=note, fillcolor=white]}}
       
       // est.py
       subgraph cluster_module_est {
          label = "est/";
      fontcolor = "#a80000";
       color = "dimgray";
       fillcolor = none;

       subgraph cluster_module_est {
          label = "kf/";
      fontcolor = "#a80000";
       color = "dimgray";
       fillcolor = none;
       
	 subgraph cluster_module_ekf {
         label = "ekf.py";
         style = rounded;
         fontcolor = "#a80000";
         EKF [label="EKF", shape=note, fillcolor=white];
       }

	 subgraph cluster_module_ukf {
         label = "ukf.py";
         style = rounded;
         fontcolor = "#a80000";
         UKF [label="UKF", shape=note, fillcolor=white];
       }       
     }}
     Observer -> System [style=dotted];
     Controller -> System [style=dotted];
     
     EKF -> Observer [style=dotted];
     UKF -> Observer [style=dotted];
     PID -> Controller [style=dotted];

     Observability -> System [style=dotted];
     Controllability -> System [style=dotted]; 
   }}


..  toctree::
   :maxdepth: 2
   :caption: Base pykal

   base_pykal_rationale
   
   
With ROS
^^^^^^^^

.. graphviz::
   :caption: Class Inheritance and Module Usage

   digraph ClassInheritance {
     node [shape=box, fontname="Helvetica", style=filled, fillcolor=lightgray];
     rankdir=RL;
     splines=ortho;

     // pykal/ supercluster
     subgraph cluster_pykal {
       label = "pykal/";
       fontcolor = "#a80000";
       style = dotted;
       color = black;
       fillcolor = none;

       // system.py
       subgraph cluster_system {
         label = "system.py";
         style = rounded;
         fontcolor = "#a80000";
         System [label="System", shape=note, fillcolor=lightblue];
       }

       // observer.py
       subgraph cluster_observer {
         label = "observer.py";
         style = rounded;
         fontcolor = "#a80000";
         Observer [label="Observer", shape=note, fillcolor=lightblue];
       }}

     // ros/ supercluster
     subgraph cluster_ros {
       label = "pykal/ros/";
       fontcolor = "#a80000";
       style = dotted;
       color = black;
       fillcolor = none;

       // generate_observer_node.py
       subgraph cluster_generate_observer_node {
         label = "generate_observer_node.py";
         style = rounded;
         fontcolor = "#a80000";
         ObserverNode [label="ObserverNode", shape=note, fillcolor=lightblue];
       }

       // generate_topic2sys_node.py
       subgraph cluster_generate_topic2sys_node {
         label = "generate_topic2sys_node.py";
         style = rounded;
         fontcolor = "#a80000";
         Topic2SysNode [label="Topic2SysNode", shape=note, fillcolor=lightblue];
       }

       // utils/ subcluster
       subgraph cluster_utils_ros {
         label = "utils/";
         fontcolor = "#a80000";
         style = dotted;
         color = black;
         fillcolor = none;

         // systems.py
         subgraph cluster_systems {
           label = "systems.py";
           style = rounded;
           fontcolor = "#a80000";
           Turtlebot [label="Turtlebot", shape=note, fillcolor=white];
         }

         // topic2sys.py
         subgraph cluster_topic2sys {
           label = "topic2sys.py";
           style = rounded;
           fontcolor = "#a80000";
           odom2pose [label="odom2pose", shape=note, fillcolor=white];
           pose2odom [label="pose2odom", shape=note, fillcolor=white];
         }
       }
     }     


     // Usage Relationships
     ObserverNode -> Turtlebot [style=dotted, label="uses"];
     ObserverNode -> Observer [style=dotted, label="uses"];
     Turtlebot -> System [style=dotted, label="uses"];     
     ObserverNode -> Topic2SysNode [style=dotted, label="uses"];
     Topic2SysNode -> odom2pose [style=dotted, label="uses"];
     Topic2SysNode -> pose2odom [style=dotted, label="uses"];     
   }


Node and Topic Topology
^^^^^^^^^^^^^^^^^^^^^^^

.. graphviz::
   :caption: Node and Topic Topology

   digraph NodeTopicTopology {
     node [shape=box, fontname="Helvetica", style=filled, fillcolor=lightgray];
     rankdir=LR;
     splines=ortho;

     // ros/ supercluster
     subgraph cluster_topic_override {
       label = "topic_override_module";
       fontcolor = "#0044cc";
       style = dotted;
       color = black;
       fillcolor = none;
     


         topic2sys [label="/topic2sys", shape=ellipse, style=filled, fillcolor=white];

	 observer [label="/observer", shape=ellipse, style=filled, fillcolor=white];
       

         sys_meas [label="/sys_meas", shape=box, style=filled, fillcolor=white];
         sys_state_est [label="/sys_state_est", shape=box, style=filled, fillcolor=white];
	 }
	 renamed_topic [label="/renamed_topic", shape=box, style=filled, fillcolor=lightblue, fontcolor="#0044cc"];
	 topic [label="/topic", shape=box, style=filled, fillcolor=lightblue, fontcolor="#0044cc"];
	 


     topic2sys -> sys_meas; 
     topic2sys -> topic;
     sys_state_est -> topic2sys;
     renamed_topic -> topic2sys;
     sys_meas -> observer;
     observer -> sys_state_est;
   }
