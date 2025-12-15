================
Getting Started
================

``pykal`` is a Python package that serves as a bridge between theoretical control/estimation algorithms and their implementation onto hardware. It does this by treating the discrete-time dynamical system as a compositional primitive with which we may implement any algorithm we like. 

This tutorial walks through the entirety of the ``pykal`` API using simple example systems, where we begin with block diagrams and end with robust ROS node architectures that may be deployed as needed. 

Although I often find it useful, in learning a new tool, to first understand its history (e.g. in what context was it created, what problems was it built to solve), the reader may skip the Background section and  begin with the "Quickstart: pykal Workflow" section without any loss in utility. Similarly, the reader can skip the "Installation" section if they already installed pykal, ros 2, and gazebo on their computer. 

..  toctree::
    :caption: Contents
    :maxdepth: 2

    background
    installation
    ../notebooks/pykal_workflow
    ../notebooks/curse_of_hardware
    ../notebooks/simulating_the_curse
    ../notebooks/ros_deployment
    ../notebooks/ros_deployment_and_gazebo
