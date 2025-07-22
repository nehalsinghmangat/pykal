==========
Quickstart
==========
This section of the tutorial introduces how to model a control system using ``pykal_core``. This sections assumes a minimal background in control; in particular, the concept of systems, signals, observers, and controllers. For such a minimal working background, please consult the conceptual reference here:


We show below a series of control systems, starting with the canonical pendulum and ending with the classic problem of balancing an inverted pendulum. We represent systems, signals, observers, and controllers as instances of ``System``, ``Observer``, ``Signal`` and ``Controller`` classes, respectively. A Jupyter notebook containing all of the code in the following sections, as well as additional material, can be viewed and downloaded here:

.. toctree::
   :maxdepth: 2
   :caption: Notebook

   ../notebooks/quickstart

Dynamical Systems
-----------------
We consider a pendulum.

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: space-between; gap: 2em; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/system_block_diagram.svg" alt="System Diagram" style="width: 40%;">
     <img src="../../_static/pendulum.svg" alt="Pendulum Diagram" style="width: 30%;">
     <img src="../../_static/pendulum_state_space.svg" alt="Pendulum State Space" style="width: 30%;">     
   </div>


.. code-block:: python

   from pykal_core.control_system.system import System
   from numpy.typing import NDArray
   import numpy as np

   def f_simple_dynamics(xk: NDArray) -> NDArray:  # k = m = 1
       A = np.array([[0, 1],
                     [-1, 0]])  
       return A @ xk

   def h_position(xk: NDArray) -> NDArray:
       return np.atleast_2d(xk[0])

   sys = System(
       state_names=['x0', 'x1'],
       measurement_names=["x0_meas"],
       f=f_simple_dynamics,
       h=h_position
   )
	   
Bind Behaviour, not Data
^^^^^^^^^^^^^^^^^^^^^^^^
Note that the System constructor initializes the sys object with functions f and h. In pykal_core, with very few exceptios, objects bind behaviour (functions), not data (e.g. initial conditions, simulation results, etc...; note that we do bind metadata such as state_names and measurement_names). This statelessness is a key feature of pykal_core, as it allows us to build control systems of arbitrary complexity without worrying about shared mutable state.

Signals
-------
Consider a pendulum to which we apply a sinusoidal torque.

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: space-between; gap: 2em; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/system_input_block_diagram.svg" alt="Signal System Diagram" style="width: 40%;">
     <img src="../../_static/pendulum_torque.svg" alt="Pendulum Torque Diagram" style="width: 20%;">
     <img src="../../_static/pendulum_torque_state_space.svg" alt="Pendulum State Space" style="width: 40%;">     
   </div>

Building Functions with Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Note the use of a factory function to define a signal function. In pykal_core, whenever possible, functions are used in place of classes (e.g. a factory function that creates a sinusoidal function as opposed to some Sinusoidal class that instantiates a sin object with a signal method). This is for the sake of simplicity and, as we will see in the next section, ease in composing multiple signals. 

Composing Signals
-----------------
.. raw:: html

   <div style="display: flex; justify-content: center; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/system_input_multiple_block_diagram.svg" 
          alt="Multiple Signal System Diagram" 
          style="width: 60%;">
   </div>

.. raw:: html

   <div style="display: flex; justify-content: center; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/signals_combined.svg" 
          alt="Signal Composition" 
          style="width: 100%;">
   </div>

Composing Functions with Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At first glance, the signal.transform method may seem strange, as the arguments (a,b) to which our function (add) is applied are themselves functions and not, say, numpy arrays. This is because signal.transform does not return a transformed numpy array, but rather a new function (sig) which returns the transformed numpy array given the transformation and the functions we are transforming. This is an incredibly powerful paradigm, because we can now construct signals transformations of arbitrary complexity all in a single function, which makes representing said functin in  ros node much easier. Also, the sigmal need not be signal functions; it could be functions that return measurements (such as h) or estimated state (in the case of observers, shown next). This signal class does a lot of heavy lifting. 


Observers
---------

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: space-between; gap: 2em; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/system_input_observer_block_diagram.svg" alt="System Observer Block Diagram" style="width: 70%;">
     <img src="../../_static/pendulum_torque_laser.svg" alt="Pendulum with Laser Sensor" style="width: 30%;">
   </div>

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: center; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/pendulum_torque_laser_state_space.svg" alt="Pendulum Laser State-Space Equations" style="width: 40%;">
   </div>

Functions without Side-effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The observer implemented above uses a standard kalman filter, which is initialized with an initial state and then run after the fact with measurement data to update a covariance matrix to experiement. You may wonder why the covariance matrix is returned at all, if it is only the state estimate we are concerned with. In `pykal_core`, with very few expections, all functions and methods are **pure**; that is, they have no side-effects. Combined with the statelessness of pykal_core's classes (e.g. there is no obs.kf.P_hist attribute), this means much safety. It also means multiple observer instances can run on the same system at once with no fear of collisions, which of course transfers well to the ROS architecture.


Controllers
-----------

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: space-between; gap: 2em; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/system_input_observer_controller_block_diagram.svg" alt="System Observer Controller Block Diagram" style="width: 70%;">
     <img src="../../_static/pendulum_inverted_controller.svg" alt="Pendulum Inverted" style="width: 30%;">
   </div>

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: center; margin-top: 1em; margin-bottom: 1em;">
     <img src="../../_static/pendulum_inverted_controller_state_space.svg" alt="Pendulum Inverted Equations" style="width: 40%;">
   </div>   

Runtime Coupling
^^^^^^^^^^^^^^^^
We have now officially created a Feedback system (help yourself to a beer; if you are underage, then help yourself to a small beer). Note that, unlike other system modeling frameworks like MATLAB and SIMULINK, this feedback system is not actually a system in memory. There is no feedback system class, nor any object we can call that has all of these properties. This is because, in pykal, everything is a function, and we only have a system at runtime. This is known as dynamic coupling, and is a Python featrue that gives us much flexibility6 and power. The control system only exists if we simulate it, or if we call it; otherwise its just a bunch of composed functions. If we really want all of this in an abstracted away system, for example if we were interfacing two or more control systems, then we can create "system" functions which takes as an input blank and outputs. Of course, such "system" functions can then be composed with other "system" functions, and even saved in memory to be called later (in Python, functions are instances of the Function class, so we can save this function in memory). Calling these is shown later in the notebook.

