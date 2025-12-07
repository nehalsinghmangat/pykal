==================
Theory to Software
==================

We consider for illustrative purposes a simple control system. Consider the thermostat. When you adjust the temperature on the thermostat, you are modifying the **setpoint generator**, which generates the reference temperature the thermostat attempts to maintain.

We may model this setpoint generator as a **discrete-time dynamical system**.

Let:

- :math:`u_k` be the discrete input at time step :math:`k`, taking values in 
  :math:`\{ -1, 0, +1 \}` (corresponding to pressing the "down" or "up" buttons, with 0 being "no signal"),
- :math:`x_k` be the state, representing the current reference temperature.

The evolution equation (state update) is then:

.. math::

    x_{k+1} = f(x_k, u_k) = x_k + u_k.

The output of the system is simply the current reference temperature.  
Thus the output map is the identity:

.. math::

    y_k = h(x_k) = x_k.

------------------------------------
Implementing the Model in Software
------------------------------------

Using *pykal*’s :class:`~pykal.DynamicalSystem` class, we can represent this same system directly
in Python:

.. code-block:: python

    import pykal

    def f(ref_temp, uk):
        return ref_temp + uk

    def h(ref_temp):
        return ref_temp

    setpoint_generator = pykal.DynamicalSystem(
        f=f,
        h=h,
        state_name="ref_temp"
    )

This defines the setpoint generator as a software dynamical system.  
Note that the identifier ``state_name`` **must match** the name of the state variable used in
the function signatures. This ensures that the internal dispatcher in *pykal* can correctly bind
function arguments during execution.

---------------------------------------
Simulating the Setpoint Generator
---------------------------------------

Once the dynamical system has been instantiated, we may simulate its evolution using the
:meth:`~pykal.DynamicalSystem.step` method. The ``step`` method receives a dictionary of input
arguments (matching the function signatures of ``f`` and ``h``) and returns either the output
alone or the pair ``(next_state, output)`` depending on the specified flags.

For our thermostat example, a single time-step update can be carried out as follows:

.. code-block:: python

    # Initial state
    ref_temp = 20.0      # degrees Celsius

    # Example user inputs across several time steps
    user_inputs = [0, +1, +1, 0, -1, 0]

    history = [ref_temp]

    for uk in user_inputs:
        # Advance the dynamical system by one step
        next_state, yk = setpoint_generator.step(
            return_state=True,
            param_dict={"ref_temp": ref_temp, "uk": uk}
        )

        # Record output and update state variable
        history.append(yk)
        ref_temp = next_state

    print(history)

The list ``history`` contains the reference temperature across each time step. Observe that the
entire control-theoretic model is implemented using only the evolution function, the output
function, and the structured dispatch rules provided by the *pykal* framework.

In general, any discrete-time control or estimation subsystem may be encoded in precisely this
manner: a pair ``(f, h)`` together with a consistent state identifier. This structure allows
complex multi-block systems to be assembled compositionally using *pykal*’s higher-level tools.

===========================================
A Closed-Loop Thermal Control System in pykal
===========================================

We now extend the thermostat example to a full closed-loop control system.  
The elements are:

- a **plant**: the room temperature dynamics,
- a **sensor**: noisy temperature measurements,
- an **observer**: a discrete-time Kalman filter for state estimation,
- a **controller**: a PID controller that drives the estimated temperature to the reference.

All components are implemented as dynamical systems in *pykal*.

-----------------
Plant: Temperature
-----------------

We model the room temperature :math:`T_k` (in degrees Celsius) as a first-order system with
thermal time constant :math:`\tau`, environmental temperature :math:`T_{\text{env}}`, and a
heater whose input :math:`u_k \in [0, 1]` scales the heating power:

.. math::

   T_{k+1}
   = T_k + \frac{\Delta t}{\tau}
     \Big( - (T_k - T_{\text{env}}) + K_{\text{heater}} u_k \Big).

The output of the plant is simply the (true) temperature:

.. math::

   y_k = h(T_k) = T_k.

In *pykal*, we represent this as:

.. code-block:: python

   import numpy as np
   import pykal

   def f_plant(T, u, dt, tau, K_heater, T_env):
       """Discrete-time thermal plant."""
       return T + (dt / tau) * (-(T - T_env) + K_heater * u)

   def h_plant(T):
       """Plant output is the true temperature."""
       return T

   plant = pykal.DynamicalSystem(
       f=f_plant,
       h=h_plant,
       state_name="T",
   )

-----------------------------
Observer: Scalar Kalman Filter
-----------------------------

We assume a scalar linear model around an operating point:

.. math::

   x_{k+1} &= a x_k + b u_k + w_k, \\
   y_k &= c x_k + v_k,

where :math:`x_k` is the temperature state, :math:`y_k` is the noisy measurement,
and :math:`w_k \sim \mathcal{N}(0, Q)` and :math:`v_k \sim \mathcal{N}(0, R)`.

A scalar discrete-time Kalman filter maintains the estimate :math:`\hat{x}_k` and variance :math:`P_k`.
The prediction and update steps are:

.. math::

   \text{Prediction:} \quad
      \hat{x}_{k|k-1} = a \hat{x}_{k-1} + b u_{k-1}, \quad
      P_{k|k-1} = a^2 P_{k-1} + Q,

.. math::

   \text{Update:} \quad
      S_k = c^2 P_{k|k-1} + R, \quad
      K_k = \frac{P_{k|k-1} c}{S_k}, \\
      \hat{x}_k = \hat{x}_{k|k-1} + K_k (y_k - c \hat{x}_{k|k-1}), \quad
      P_k = (1 - K_k c) P_{k|k-1}.

We store :math:`(\hat{x}_k, P_k)` together as the Kalman filter “state” and use the estimate
as the observer output. Implemented as a dynamical system:

.. code-block:: python

   def f_kf(xhat_P, y, u, a, b, c, Q, R):
       """
       Scalar discrete-time Kalman filter state update.

       xhat_P : np.ndarray of shape (2,)
           [x_hat, P]
       """
       x_hat, P = xhat_P

       # Prediction
       x_pred = a * x_hat + b * u
       P_pred = a * P * a + Q

       # Innovation
       S = c * P_pred * c + R
       K = P_pred * c / S

       # Update
       x_new = x_pred + K * (y - c * x_pred)
       P_new = (1.0 - K * c) * P_pred

       return np.array([x_new, P_new], dtype=float)

   def h_kf(xhat_P):
       """Observer output is the estimated state x_hat."""
       x_hat, P = xhat_P
       return x_hat

   kf = pykal.DynamicalSystem(
       f=f_kf,
       h=h_kf,
       state_name="xhat_P",
   )

-----------------------
Controller: PID in pykal
-----------------------

We adopt a standard PID controller acting on the **estimated** temperature :math:`\hat{x}_k`:

.. math::

   e_k &= r_k - \hat{x}_k, \\
   I_k &= I_{k-1} + e_k \Delta t, \\
   D_k &= \frac{e_k - e_{k-1}}{\Delta t}, \\
   u_k &= K_p e_k + K_i I_k + K_d D_k.

We take the controller state to be :math:`z_k = (I_k, e_{k-1})`. The PID can be encoded as:

.. code-block:: python

   def f_pid(z, r, x_hat, dt, Kp, Ki, Kd):
       """
       PID controller state update.

       z : np.ndarray of shape (2,)
           [I_k, e_{k-1}]
       """
       I_prev, e_prev = z

       e = r - x_hat
       I_new = I_prev + e * dt
       # We store the current error as the new e_prev for the next step.
       return np.array([I_new, e], dtype=float)

   def h_pid(z, r, x_hat, dt, Kp, Ki, Kd):
       """
       PID output (control signal) computed from current state and error.
       """
       I_k, e_prev = z
       e = r - x_hat
       D = (e - e_prev) / dt
       u = Kp * e + Ki * I_k + Kd * D
       return u

   pid = pykal.DynamicalSystem(
       f=f_pid,
       h=h_pid,
       state_name="z",
   )

-----------------------------------------
Assembling the Closed-Loop Control System
-----------------------------------------

We now combine all components into a single simulation loop:

- **Setpoint** :math:`r` is fixed (or could be time-varying),
- **Plant state** :math:`T_k` evolves according to the thermal dynamics,
- **Sensor** provides noisy measurements :math:`y_k`,
- **Kalman filter** produces :math:`\hat{x}_k`,
- **PID controller** produces the heater input :math:`u_k`,
- **Plant** consumes :math:`u_k`.

.. code-block:: python

   import numpy as np

   # --- Simulation parameters ---
   dt = 1.0              # [s]
   T_env = 20.0          # ambient temperature [°C]
   tau = 100.0           # thermal time constant [s]
   K_heater = 5.0        # heater gain
   r = 22.0              # desired temperature [°C]

   # Linearized scalar model parameters for KF
   a = 1.0 - dt / tau
   b = (dt / tau) * K_heater
   c = 1.0               # measurement is temperature itself
   Q = 0.01              # process noise variance
   R = 0.25              # measurement noise variance

   # PID gains
   Kp, Ki, Kd = 2.0, 0.1, 0.0

   # --- Initial conditions ---
   T = 20.0                       # true temperature
   xhat_P = np.array([20.0, 1.0]) # [x_hat, P]
   z_pid = np.array([0.0, 0.0])   # [I_0, e_{-1}]
   u = 0.0                        # initial heater command

   # --- Logging ---
   n_steps = 300
   T_hist = []
   xhat_hist = []
   u_hist = []
   r_hist = []

   rng = np.random.default_rng(seed=0)

   for k in range(n_steps):

       # 1) Sensor: noisy measurement of true temperature
       y = T + rng.normal(0.0, np.sqrt(R))

       # 2) Observer: Kalman filter update
       xhat_P, x_hat = kf.step(
           return_state=True,
           param_dict={
               "xhat_P": xhat_P,
               "y": y,
               "u": u,
               "a": a,
               "b": b,
               "c": c,
               "Q": Q,
               "R": R,
           },
       )

       # 3) Controller: PID acts on estimated temperature
       z_pid, u = pid.step(
           return_state=True,
           param_dict={
               "z": z_pid,
               "r": r,
               "x_hat": x_hat,
               "dt": dt,
               "Kp": Kp,
               "Ki": Ki,
               "Kd": Kd,
           },
       )

       # Optionally saturate the control to [0, 1]
       u = float(np.clip(u, 0.0, 1.0))

       # 4) Plant: propagate true temperature
       T, y_true = plant.step(
           return_state=True,
           param_dict={
               "T": T,
               "u": u,
               "dt": dt,
               "tau": tau,
               "K_heater": K_heater,
               "T_env": T_env,
           },
       )

       # 5) Log
       T_hist.append(T)
       xhat_hist.append(x_hat)
       u_hist.append(u)
       r_hist.append(r)

   # At this point T_hist and xhat_hist contain the true and estimated
   # temperatures, and u_hist contains the PID commands.

You can then visualize the performance of the closed-loop system, for example:

.. code-block:: python

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(n_steps) * dt

plt.figure()
plt.plot(t, T_hist, label="True temperature $T_k$")
plt.plot(t, xhat_hist, label="KF estimate $\hat{x}_k$", linestyle="--")
plt.plot(t, r_hist, label="Reference $r$", linestyle=":")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.legend()
plt.grid(True)
plt.show()

This example demonstrates the core philosophy of pykal: each conceptual block of a control
system—plant, observer, and controller—is represented as a dynamical system with an evolution
function :math:f and an output function :math:h, and these blocks are composed at the
Python level to realize full closed-loop behavior.   

:doc:`← The pykal pipeline <the_pykal_pipeline>` | :doc:`Software to Simulation → <composing_dynamical_systems>`
