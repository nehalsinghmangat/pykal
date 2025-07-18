================
Building a Robot
================

Building a robot is impossible. Indeed, it is so impossible it almost seems easy. When we see a humanoid jump from one platform to another, or a mechanical dog carry a hundred pounds on its back, we are impressed only a moment before we ask why the humanoid can't do a backflip or the dog recognize faces. Surely the scientists and engineers who designed these systems could have spent an extra hour or so to implement these features, no?

The more complex a system is, the more complete our abstraction of it and, paradoxically, the simpler it seems. This phenomena is not unique to robots: even those of us who are technically literate often forget the computer behind the screen or the engine under the hood. We are fortunate that such simplifications rarely, if ever, pose a problem in our lives. But there is one situation in which it is always troublesome: when we want to build a complex system ourselves.

Suppose we want to build a robot.  A few scenarious come to mind: if we were hobbyists, we might want a weekend project that could also clean our carpet; if we were engineers, we might want a mechanical arm to boost production on the assembly line; if we were academics, we might dream of a quadruped with a cannon-head to threaten the government for grant money. Regardless of the context, succesffully building a robot requires the following:

**Theory**. Robotics is a broad field and theory an even broader term, so, to be precise, when we say theory we mean "control theory". Before we can build anything more complicated than a paperclip, we need to recast our robot as a control system.

**Software**. Once we have a control system, we need to model it in software. MATLAB has been an industry standard for years; in particular, Simulink, a MATLAB extension, offers users the ability to implement systems as a block diagram using a GUI. However, MATLAB is closed source and behind a considerable pay wall; furthermore, it does not integrate well with open-source robotics platforms like ROS. Thus, programming languages such as Python or C++ have become increasingly popular options for implementing our systems *in silica*.

**Simulation**. The world of simulation is beautiful and forgiving. Here we can test the behaviour of our system under various external conditions, or even experiment with our system itself, to see, for example, how changes in the control input or system dynamics can affect performance. Crafting good simulations is an art that requires knowledge of the operating environments the system will be in and the relevant factors to consider.

**Hardware**. This is the most important step; arguably, everything until now has been an intellectual excercise. The stakes are real. When a simulation fails, it's an inconvenience. When hardware fails, it's a tragedy.

Each of the above can be difficult in their own right, but it is the bridge between them that makes building a robot impossible. Many wonderful ideas never leave the whiteboard when we lack the time or skillset to implement them, and most of the lucky few that make it to software and simulation die on the road to hardware.

This is what ``pykal`` aims to fix. 



ROS Nodes in ``pykal``
----------------------

Example: Turtlebot
------------------
Download the notebook here and have some fun!
