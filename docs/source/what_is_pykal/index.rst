==========
Quickstart
==========

``pykal`` is a Python package that serves as a bridge between theoretical control/estimation algorithms and their implementation onto hardware. It does this by treating the discrete-time dynamical system as a compositional primitive with which we may implement any algorithm we like [see full derivation here]. This induces a simple software API, which in turn makes wrapping such implementations in ROS nodes, and thus porting them onto hardware, remarkably easy.

This tutorial walks through the entirety of the ``pykal`` API using a series of example feedback systems, where we begin with a purely theoretical system and end with a robust ROS node architecture that may be deployed as needed. 

The reader may skip the background section. Although I often find it useful, in learning a new tool, to first understand its history (e.g. in what context was it created, what problems was it built to solve), the reader may begin with the "Quickstart: pykal Workflow" section without any loss in utility. 

..  toctree::
    :caption: Contents
    :maxdepth: 3

    background
    ../notebooks/pykal_workflow
    ../notebooks/curse_of_hardware
    ../notebooks/simulating_the_curse
    ../notebooks/ros_deployment
    ../notebooks/ros_deployment_and_gazebo    
