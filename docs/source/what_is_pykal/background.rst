============
 Background
============
In this section, we will discuss the personal background of the auther (insofar as it informs the form and functionality of the ``pykal`` package), the problem domain that ``pykal`` aims to solve, and the overall architecture of the package. 

Who am I?
=========
My name is Nehal Mangat, and I am a (handsome) graduate student at the University of Nevada, Reno, where I am pursuing a masters in Mechanical Engineering and Applied Mathematics. I am also a research assistant in the van Breugel roboticcs lab, where my work entails writing new state estimatoin algorithms for autonomous drones. I have also interned under the Air Force Researrch Laboratory, and currently collaboarate on algorithm design. I have been involved in teh world of academic roboticsc for several years now. I tend to take a more "mathematical" approach to my work (this will become important later on). 

I love my field, but it is a shame to say that there is something rotten in the state of robotics. This will be explained below. 

The Problem
===========
As of late, we have witnessed an explosion of control and estimation algorithms in the world of robotics. Ideally, theory becomes practice with enough iteration; unfortunately, this has not been the case in academia. Academia values algorithms that are inspired, not implemented, which leads researchers, already criminally busy, to often publish their results and shrug the hardware implementation onto future researchers-- who, in turn, make a few modificatoins, publish a result, and pass the algorithm on. Inveitably, there comes a day when the latest iteration is put onto hardware (often by some poor grad student) and the algorithm fails -- at which point, who among the twenty authors scattered across seven papers knows where the break in the chain occurred?

This is a tragedy, to say nothing of the lost productivity. It reminds me of a poignant saying, paraphrased form "We build castles upon cobwebs and are surprised we have no kingdom."

Hence, through by a collaboration between the van Breugel lab and the REEF, we built pykal. 


 The Solution
==============
Broadly speaking, building a robot occurs in four steps, and ``pykal``  strives to make every step as painless as possible:






In the following sections, we will cover

- the pykal workflow and API
- examples of casting algorithms as dynamical systems using pykal
- wrapping and running said dynamical systems in ROS nodes
- the challenges of interfacing with hardware, and how to handle them

By the end of the tutorial, you will be able to implement any of the algorithms in the Algorithm Library onto a robot of your choice, or even implement an algorithm of your own. Let's get started. 

----

:doc:`← What is pykal? <../index>` | :doc:`The pykal Workflow → <../notebooks/pykal_workflow>`

----
