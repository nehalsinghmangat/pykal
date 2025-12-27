:doc:`← Getting Started <../index>`
     
========
Overview
========
In this section, we will discuss:

- the personal background of the auther (insofar as it informs the form and function of the ``pykal`` package)
- the problem in robotics that ``pykal`` aims to solve
- the proposed solution (hint: it may be the ``pykal`` package)
- the structure of the Tutorial and how best to work through it 

Who am I?
=========
My name is Nehal Mangat, and I am a (handsome) graduate student at the University of Nevada, Reno, where I am pursuing a masters in Mechanical Engineering and Applied Mathematics. I am also a robotics research assistant in the van Breugel lab and a former intern of the Air Force Research Laboratory's Autonomous Vehicles Laboratory.

I have been involved in the world of academic robotics for several years now. I love my field, but it is a shame to say that there is something rotten in the state of robotics. 

The Problem
===========
As of late, we have witnessed an explosion of control and estimation algorithms. This should be a good thing; certainly we should not be worse off for having more ideas.  

Unfortunately, academia values novelty, not utility; that is, researchers often scurry to publish novel results at the expense of useability. Since there is little incentive to test results in reality, the hardware implementation of an algorithm is passed onto future researchers -- who, pressed to publish novel results of their own, instead extend the algorithm, publish, and pass the buck along to future researchers, *ad infinitum*.

When the day comes that the algorithm is finally implemented onto hardware, and it does not work as expected (or at all), where do we begin troubleshooting? Is it a hardware issue, a software issue, or does the real issue lie in the third paper of this algorithm's journey, where the substitution of a new numerical method lead to unforeseen consequences when faced with asynchronous sensor data? 

This is a tragedy of efficiency, to say nothing of the sanity lost in troubleshooting such issues.


The Solution
============
The ``pykal`` package is a Python package designed to make this path:

.. container:: graphviz-tight

   .. graphviz::
      :align: center

      digraph GettingStartedFlow {
          rankdir=LR;
          bgcolor="transparent";
          splines=ortho;

          node [
            shape=box,
            style="rounded,filled",
            fillcolor="white",
            color="black",
            fontname="Helvetica",
            fontsize=12,
            margin="0.12,0.08"
          ];

          edge [
            color="black",
            penwidth=1.5,
            arrowsize=0.8
          ];

          Theory [label="Theory"];
          Python [label="Python"];
          ROS    [label="ROS"];
          Gazebo [label="Gazebo"];

          Theory -> Python -> ROS -> Gazebo;
      }


   
as painless as possible. While each island can be a struggle in itself, it is on the bridges between them that good ideas die.  ``ROS`` and ``Gazebo`` are mature robotics platforms that have already eased the burden of the last bridge; ``pykal`` aims to ease the burden of the first two. How?

``pykal`` uses the discrete-time dynamical system as a compositional primitive; that is, dynamical systems are the building blocks with which we implement our algorithms. Such representations of algorithms are mathematically equivalent, and so this spans the gap between Theory and Python.

Further, such a simple compositional primitive induces a simple software API with ``ROS``, enabling the creation of wrappers that cast the ``ROS`` node architecture as networks of dynamical systems. Thus is spanned the second bridge. 

The result is a pipeline from "head to hardware" that spares painful debugging on the road to reality.

Using the Tutorial
==================
The "Tutorial" is split into four sections.

Overview
^^^^^^^^
If you have somehow managed to skip to the end of this one-page overview and are now reading this, I recommend jumping back to the beginning and reading the full overview.

Theory to Python
^^^^^^^^^^^^^^^^
If you are new to this package and want to try casting algorithms/control systems as dynamical systems, start here. 

Python to ROS
^^^^^^^^^^^^^
If you are comfortable using ``pykal`` to cast algorithms/control systems as dynamical systems and want to begin simulating things in ``ROS``, start here.

ROS to Gazebo
^^^^^^^^^^^^^
If you are comfortable simulating things in ``ROS`` and want to put it in ``Gazebo`` to see how it interacts with the world, start here. 


:doc:`← Getting Started <./index>` 
