.. graphviz::
   :caption: Class Inheritance and Module Usage

   digraph ClassInheritance {
       node [shape=box, fontname="Helvetica", style=filled, fillcolor=lightgray];
       rankdir=LR;

       // Base classes
       subgraph cluster_base {
           label = "_base_kf.py";
           style = rounded;
	   fontcolor = "#a80000";
           BaseKF [label="BaseKF\n(Abstract)", fillcolor=lightgray];
       }

       // kf.py
       subgraph cluster_kf {
           label = "kf.py";
           style = rounded;
	   fontcolor = "#a80000";
           EKF [label="EKF", fillcolor=lightblue];
           UKF [label="UKF", fillcolor=lightblue];
       }

       // system.py
       subgraph cluster_system {
           label = "system.py";
           style = rounded;
	   fontcolor = "#a80000";
           System [label="System", shape=note, fillcolor=lightblue];
       }

       // safeio.py
       subgraph cluster_safeio {
           label = "safeio.py";
           style = rounded;
	   fontcolor = "#a80000";
           SafeIO [label="Safeio", shape=note, fillcolor=white];
       }

       // Inheritance relationships
       EKF -> BaseKF;
       UKF -> BaseKF;
       System -> SafeIO;
       
       // Usage relationships
       BaseKF -> System [style=dotted, label="uses"];

   }
