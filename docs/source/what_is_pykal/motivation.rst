==========
Motivation
==========

I often find it useful, when handed a new tool, to turn it over in my hands and ask what it is and why it was created. pykal is a Python package for casting algorithms as dynamical systems and running said systems in ROS. This is the what, and before you turn it over, I would like to tell you the why. 

As of late, we have witnessed an explosion of control and estimation algorithms in the world of robotics. Ideally, theory becomes practice with enough iteration; unfortunately, evolution is crippled in the world of academics. Academia values inspiration, not implementation; researchers often pass the buck of implementating their algorithms onto future researchers, who in turn modify it and pass it on to future researchers, until one day an algorithm is finally put onto hardware and it fails -- at which point, who among the twenty authors scattered across seven papers knows where the break in the chain occurred?

pykal was built to put a stop this. In this tutorial, we will cover

- the pykal workflow and API
- examples of casting algorithms as dynamical systems using pykal
- wrapping and running said dynamical systems in ROS nodes
- the challenges of interfacing with hardware, and how to handle them

By the end of the tutorial, you will be able to implement any of the algorithms in the Algorithm Library onto a robot of your choice, or even implement an algorithm of your own. Let's get started. 

----

:doc:`← What is pykal? <../index>` | :doc:`The pykal Workflow → <../notebooks/pykal_workflow>`

----
