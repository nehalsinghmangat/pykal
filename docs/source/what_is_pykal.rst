What is pykal?
--------------
``pykal`` is a framework to get ideas from the page to production. The package is written in python to encourage rapid prototyping and experimentation, although much effort has been made to restrict the frustrations that come with a strongly dynaimcally-typed language (see System here). 

From Theory to Software
-----------------------


We represent systems, observers, and controllers as instances of a ``System``, ``Observer``, and ``Controller`` class, respectively. For example, consider the following open-loop control system that is the mighty toaster, and the relationship between instances of ``System`` and ``Controller``:


If we have a closed-loop system, say a smart toaster that gets the bread right everytime, then we include an ``Observer`` for feedback.


In this way, systems of arbitrary complexity can be implemented in software. For more information on how the ``System``, ``Observer``, and ``Controller`` classes work together, please reference the following:

From Software to Simulation
---------------------------
This section assumes familiary with ROS (link). If such a familiarity is lacking, please check out the official website and guide.  


From Simulation to Hardware
---------------------------
