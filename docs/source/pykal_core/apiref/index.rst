API Reference
=============

API Reference
=============

.. graphviz::
   :caption: pykal_core Class and Module Structure
   :align: center

   digraph ClassStructure {
       node [shape=box, fontname="Helvetica", style=filled, fillcolor=lightgray];
       rankdir=TB;
       edge [arrowhead=normal, arrowsize=0.7, color=black];

       // ======= Main package cluster =======
       subgraph cluster_pykal_core {
           label = "pykal_core/";
           fontcolor = "#a80000";
           style = dotted;
           color = black;
           fillcolor = none;

           // ======= control_system subpackage =======
           subgraph cluster_control_system {
               label = "/control_system/";
               fontcolor = "#a80000";
               style = dotted;
               color = black;
               fillcolor = none;

               subgraph cluster_module_system {
                   label = "system.py";
                   fontcolor = "#a80000";
                   style = rounded;
                   color = black;
                   fillcolor = none;
		   SafeIO [label="SafeIO", fillcolor=white];
                   System [label="System", fillcolor=lightblue];
               }

               subgraph cluster_module_observer {
                   label = "observer.py";
                   fontcolor = "#a80000";
                   style = rounded;
                   color = black;
                   fillcolor = none;
                   Observer [label="Observer", fillcolor=lightblue];
               }

               subgraph cluster_module_controller {
                   label = "controller.py";
                   fontcolor = "#a80000";
                   style = rounded;
                   color = black;
                   fillcolor = none;
                   Controller [label="Controller", fillcolor=lightblue];
               }
           }

           // ======= control_system.signal split out =======
           subgraph cluster_control_system_signal {
               label = "/function_blocks/";
               fontcolor = "#a80000";
               style = dotted;
               color = black;
               fillcolor = none;

               subgraph cluster_module_signal {
                   label = "signal.py";
                   fontcolor = "#a80000";
                   style = rounded;
                   color = black;
                   fillcolor = none;
                   Signal [label="Signal", fillcolor=lightblue];
                   Generate [label="Generate", fillcolor=lightblue];
                   Transform [label="Transform", fillcolor=lightblue];	       
               }
           }

           // ======= utils cluster #1: compute.py only =======
           subgraph cluster_utils_1 {
               label = "/utils/";
               fontcolor = "#a80000";
               style = dotted;
               color = black;
               fillcolor = none;

               subgraph cluster_module_compute {
                   label = "compute.py";
                   fontcolor = "#a80000";
                   style = rounded;
                   color = black;
                   fillcolor = none;
                   Error [label="Error", fillcolor=lightblue];
                   Jacobian [label="Jacobian", fillcolor=lightblue];
                   Matrix [label="Matrix", fillcolor=lightblue];
                   Observability [label="Observability", fillcolor=lightblue];
                   Simulation [label="Simulation", fillcolor=lightblue];
               }
           }

           // ======= utils cluster #2: estimators and controllers =======
           subgraph cluster_utils_2 {
               label = "/utils/";
               fontcolor = "#a80000";
               style = dotted;
               color = black;
               fillcolor = none;


               subgraph cluster_controllers {
                   label = "/controllers/";
                   fontcolor = "#a80000";
                   style = dotted;
                   color = black;
                   fillcolor = none;

                   subgraph cluster_module_pid {
                       label = "pid.py";
                       fontcolor = "#a80000";
                       style = rounded;
                       color = black;
                       fillcolor = none;
                       pid [label="PID", fillcolor=lightblue];
                   }
               }

               subgraph cluster_estimators {
                   label = "/estimators/";
                   fontcolor = "#a80000";
                   style = dotted;
                   color = black;
                   fillcolor = none;

                   subgraph cluster_module_kf {
                       label = "kf.py";
                       fontcolor = "#a80000";
                       style = rounded;
                       color = black;
                       fillcolor = none;
                       ekf [label="EKF", fillcolor=lightblue];
                       ukf [label="UKF", fillcolor=lightblue];
                   }
               }
	       
           }
       }

       // ======= Edges =======
       SafeIO -> System;
       System -> Controller;
       System -> Observer;
       System -> Jacobian;
       System -> Observability;
       System -> Simulation;

       Signal -> Generate;
       Signal -> Transform;

       Controller -> pid;       
       Observer -> ekf;
       Observer -> ukf;

   }




..  toctree::
    :caption: Main modules 
    :maxdepth: 3

    pykal_core.control_system.system
    pykal_core.control_system.signals
    pykal_core.control_system.observer
    pykal_core.control_system.controller
    pykal_core.utils.safeio
   
   
..  toctree::
    :caption: Estimator Modules
    :maxdepth: 3
	       
    pykal_core.est.kf.ekf.rst
	 
