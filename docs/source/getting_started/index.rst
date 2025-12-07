===============
Getting Started
===============

Welcome to pykal! This section will get you up and running quickly.

Installation
============

.. code-block:: bash

   pip install pykal

Quick Start
===========

Here's a minimal example to see pykal in action:

.. code-block:: python

   from pykal import DynamicalSystem
   import numpy as np

   # Define a simple system
   def f(x, u):
       """Simple integrator dynamics"""
       return x + u * 0.1

   # Create dynamical system
   system = DynamicalSystem(
       f=f,
       state_name='x',
       param_dict={'x': np.array([0.0])},
       f_params_names=['x', 'u']
   )

   # Step the system
   u = np.array([1.0])
   observation = system.step({'u': u})
   print(f"New state: {observation}")

What's Next?
============

- **Browse the** :doc:`../algorithm_library` to find implementations you can use
- **See examples** in :doc:`../robot_examples/index` for platform-specific demos
- **Understand the concepts** in :doc:`../what_is_pykal/index`
- **Deploy to robots** via :doc:`../simulation_to_hardware/index`

Key Concepts
============

pykal bridges four stages:

1. **Theory** - Mathematical models of control systems
2. **Software** - Python implementations using DynamicalSystem
3. **Simulation** - Testing in Gazebo via ROS2 nodes
4. **Hardware** - Deployment to real robots (TurtleBot, Crazyflie)

.. seealso::

   - :doc:`../algorithm_library` - Browse implemented algorithms
   - :doc:`../what_is_pykal/the_pykal_pipeline` - Detailed pipeline explanation
