===========================
Motivation: Building Robots
===========================

Suppose we want to build a robot. If we were hobbyists, we might want a weekend project; if we were engineers, we might want a mechanical arm on the assembly line; and if we were academics, we might dream of a quadruped with a cannon-head to threaten the government for grant money.

Let us focus on this last group. Facetious threats aside, there has been a Cambrian explosion of control and estimation algorithms in the world of academic robotics. Unfortunately, evolution in the academic world is a sordid affair. Academia values novelty, not utility, and without an incentive to test their algorithms in hardware, researchers choose to pass the buck of implementation onto future researchers, who in turn pass it on to future researchers, until one day an algorithm is finally tested in hardware and it fails — and at that point, who among the thirty authors scattered across seven papers knows where in the chain the break occurred?

We are building castles upon cobwebs, and robotics deserves better.

pykal was built to bridge the gap between theory and hardware, and to do so in a way that is robust, extensible, and intuitive. Scope is a wonderful thing, and, if you'll forgive the pun, pykal is interested solely in the “soul” of the machine — that is, in the implementation of control and estimation algorithms.

In this tutorial, we will discuss, very broadly, the challenges faced in implementing algorithms in hardware; how discrete-time dynamical systems offer a clean mathematical framework for modeling algorithms; how such a mathematical model induces a correspondingly simple API between software and hardware; and finally, a full example of the pykal pipeline from theory to hardware.

----

:doc:`← What is pykal? <../index>` | :doc:`The pykal pipeline → <the_pykal_pipeline>`

----
