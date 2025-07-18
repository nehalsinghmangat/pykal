==============================
 Control Systems in ``pykal``
==============================


Dynamical Systems
^^^^^^^^^^^^^^^^^
Recall that the state-space model of a continuous-time dynamical system is given by:

.. math::

   \begin{aligned}
   \dot{x} &= f(x, u, t) + w(t) \\
   y &= h(x, u, t) + r(t)
   \end{aligned}

where :math:`x \in \mathbb{R}^{n}` is the **state vector**, :math:`u \in \mathbb{R}^{p}` is the **input vector**, :math:`t \in \mathbb{R}^{+}` is **time**, and :math:`w` is a random variable referred to as **process noise** with :math:`w(t) \in \mathbb{R}^{n}` distributed as :math:`w \sim \mathcal{N}(0, Q(t))`, where :math:`Q(t) \in \mathcal{M}_{n}(\mathbb{R})` is the positive semidefinite **process noise covariance matrix**.

Similarly, :math:`y \in \mathbb{R}^{m}` is the **measurement vector**, and :math:`r` is a random variable referred to as **measurement noise** with :math:`r(t) \in \mathbb{R}^{m}` distributed as :math:`r \sim \mathcal{N}(0, R(t))`, where :math:`R(t) \in \mathcal{M}_{m}(\mathbb{R})` is the positive semidefinite **measurement noise covariance matrix**.

Note that the pertinent parts of the equation are the functions f, h, w and r. The System class in pykal uses this abstraction  and binds related functions into a given system object. We define a basic system object below. System requires that you initialize it with some initial state names. Everything else has a default. 

.. code-block:: python
   :linenos:

   from pykal.control_system.system import System

   sys = System(state_names=['x0','x1'])

Note that sys has the following attributes:

.. code-block:: python
   :linenos:

   print(sys.__dict__)

   'safeio': <pykal.control_system.system.System.SafeIO object at 0x7cc6a4153680>,
   '_state_names': ['x0', 'x1'],
   '_measurement_names': ['x0', 'x1'],
   '_system_type': 'cti',
   '_f': <function System.f_zero at 0x7cc677d5cfe0>,
   '_h': <function System.h_identity at 0x7cc677d5eac0>,
   '_Q': <bound method System.Q_uniform of <pykal.control_system.system.System object at 0x7cc677d60bc0>>,
   '_R': <bound method System.R_uniform of <pykal.control_system.system.System object at 0x7cc677d60bc0>>

The default dynamics function is a zero derivitive vector, the default measurement model is a full-state unwieghted measurement, and the default Q and R are diagonal matrices with uniform noise distribution across states. To visualize this (admittedly boring) system, we can use the following:





Observers
^^^^^^^^^

Controllers
^^^^^^^^^^^
