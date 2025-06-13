Utilities
=========

Core helpers
------------

.. autofunction:: pykal.utils.call.call

.. autoclass:: pykal.utils.systemtype.SystemType
   :members:
   :show-inheritance:

Dynamics / Models
-----------------

.. autofunction:: pykal.utils.dynamics.linear_damped_SHO
.. autofunction:: pykal.utils.dynamics.linear_3D
.. autofunction:: pykal.utils.dynamics.cubic_damped_SHO
.. autofunction:: pykal.utils.dynamics.van_der_pol
.. autofunction:: pykal.utils.dynamics.duffing
.. autofunction:: pykal.utils.dynamics.lotka
.. autofunction:: pykal.utils.dynamics.cubic_oscillator 
.. autofunction:: pykal.utils.dynamics.rossler
.. autofunction:: pykal.utils.dynamics.hopf
.. autofunction:: pykal.utils.dynamics.lorenz
.. autofunction:: pykal.utils.dynamics.logistic_map

# control-enabled variants
.. autofunction:: pykal.utils.dynamics.van_der_pol_control
.. autofunction:: pykal.utils.dynamics.duffing_control
.. autofunction:: pykal.utils.dynamics.lotka_control
.. autofunction:: pykal.utils.dynamics.cubic_oscillator_control
.. autofunction:: pykal.utils.dynamics.rossler_control
.. autofunction:: pykal.utils.dynamics.hopf_control
.. autofunction:: pykal.utils.dynamics.lorenz_control
.. autofunction:: pykal.utils.dynamics.logistic_map_control
.. autofunction:: pykal.utils.dynamics.logistic_map_multicontrol

Measurement functions
---------------------

.. autofunction:: pykal.utils.measurements.h_identity
.. autofunction:: pykal.utils.measurements.h_linear_matrix
.. autofunction:: pykal.utils.measurements.h_partial

.. autofunction:: pykal.utils.measurements.h_identity_control
.. autofunction:: pykal.utils.measurements.h_linear_control

.. autofunction:: pykal.utils.measurements.h_range
.. autofunction:: pykal.utils.measurements.h_bearing
.. autofunction:: pykal.utils.measurements.h_range_bearing
.. autofunction:: pykal.utils.measurements.h_quadratic

.. autofunction:: pykal.utils.measurements.h_range_control
.. autofunction:: pykal.utils.measurements.h_range_bearing_control
.. autofunction:: pykal.utils.measurements.h_nonlinear_control

Input-signal generators
-----------------------

.. autofunction:: pykal.utils.controls.u_constant
.. autofunction:: pykal.utils.controls.u_sinusoidal
.. autofunction:: pykal.utils.controls.u_step
.. autofunction:: pykal.utils.controls.u_bangbang
.. autofunction:: pykal.utils.controls.u_ramp
.. autofunction:: pykal.utils.controls.u_random