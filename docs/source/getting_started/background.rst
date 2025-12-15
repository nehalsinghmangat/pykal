============
 Background
============
In this section, we will discuss the personal background of the auther (insofar as it informs the genesis of the ``pykal`` package), the problem domain that ``pykal`` aims to solve, and the solution ``pykal`` proposes. 

Who am I?
=========
My name is Nehal Mangat, and I am a (handsome) graduate student at the University of Nevada, Reno, where I am pursuing a masters in Mechanical Engineering and Applied Mathematics. I am also a robotics research assistant in the van Breugel lab and a former intern of the Air Force Research Laboratory's Autonomous Vehicles Laboratory.

I have been involved in the world of academic robotics for several years now. I love my field, but it is a shame to say that there is something rotten in the state of robotics. 

The Problem
===========
As of late, we have witnessed an explosion of control and estimation algorithms. This should be a good thing; certainly we should not be worse off for having more ideas.

Unfortunately, academia values novelty, not utility; that is, researchers, already a criminally busy bunch, often scurry to publish novel results lest they are scooped by one of their colleagues, and then shrug the hardware implementation of their algorithm onto future researchers -- who, pressed to publish novel results of their own, instead modify the algorithm slightly, push out a paper, and pass along the hardware implementation onto future researchers, and so on. 

But technical debt always comes due. When the day comes that the algorithm is finally implemented onto an actual robot, and it does not work, who among the twenty authors scattered across seven papers would know where the break in the chain occurred?

This is a tragedy, to say nothing of the lost productivity. As the saying goes: "We build castles upon cobwebs and are surprised we have no kingdom."


The Solution
============

The ``pykal`` worklow consists of the following steps:


The ``pykal`` package lives on the bridge between theory and software; the problem of the second and third bridges are solved by ``Gazebo`` and  ``ROS``. Both are mature open-source ecosystems, and so are leveraged heavily in the ``pykal`` workflow. 

How pykal works
===============
As mentioned earlier, ``pykal`` casts algorithms as the composition of discrete-time dynamical systems. This induces a simple software API, which in turn makes wrapping such implementations in ROS nodes, and thus porting them onto hardware, remarkably easy. Treating algorithms as dynamical systems is not a new idea, but using this method to effectively model **any** algorithm in software is surprisingly novel.


How to use pykal
================

In the following sections, we will cover

- examples of casting algorithms as dynamical systems
- examples of simulating noise and other data-corrupting phenomena
- wrapping our dynamical systems in ROS nodes
- simulating our system in Gazebo

By the end of the tutorial, you will be able to implement any of the algorithms in the Algorithm Library onto a robot of your choice, or even implement an algorithm of your own. Let's get started.

----

:doc:`← What is pykal? <../index>` | :doc:`The pykal Workflow → <../notebooks/pykal_workflow>`
