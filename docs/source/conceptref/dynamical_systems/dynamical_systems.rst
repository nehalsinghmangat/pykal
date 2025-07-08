What is a Dynamical System?
===========================
.. epigraph::

   "All the world's a [dynamical system],
   
   And all the men and women merely [dynamical systems]."

   -- William Shakespeare (paraphrased)

Everything. Every drop of rain or ripple in a pond; the worms in the dirt and the wires in your brain. You drive a dynamical system to work; you watch a dynamical system rise in the morning and set in the evening; and don't look now, but a pair of dynamical systems flick across the screen as you read this text. A **dynamical system** is simply a system that changes over time. This definition can conceivably apply to anything, so it depends on our discretion to use it meaningfully. A gnat is a dynamical system. A clock is a dynamical system.  We may concede that a broken clock or a dead gnat are systems, but they are dynamical no longer lest the clock should rust or the gnat decay. We often consider sufficiently complex dynamical systems to be composed of **subsystems** (e.g. every cell of a gnat is a world unto itself); conversly, we may claim our system is itself a subsystem of a larger **ecosystem** (consider how the average swamp is a buzzing of many gnats along with mosquitoes, flies, moths, and, why not, clocks.)

Modeling a Dynamical System
===========================
.. epigraph::

   "If everything is important, then nothing is."

   — Patrick Lencioni
   
Consider the pendulum. Lift it up and let it swing. The pendulum is now a dynamical system. How should we model it? For simplicity, let us assume the pendulum is not affected by air resistance; that is, the pendulum swings in a vacuum. Is this realistic? Absolutely not. But often the first step in modeling a system is to make simplifying assumptions; a simple system is a wonderful thing. So let's keep going.  The weight at the end of the rod is solid and of uniform density; the rod itself is rigid and massless; the pendulum has no internal friction and is not affected by the air conditioning in the room or the light filtering through the windows or the rotation of the earth or relativistic mass gain or...

After all is said and done, we are left with an idealized pendulum which lends itself well to a simple equation of motion:


This equation is called the dynamics of the pendulum. Now for the magic. Beta term for resistance. 



we can start reintroducing real-world considerations. For example, suppose our pendulum stood near a window in a room with air-conditioning. We brush aside all errata and focus only on the pendulum, isolated from the its environment. Going one step further, we begin to simplify aspects of the pendulum itself.; , ignorant of its own internal  the light filtering through  This will reduce our system to something that is both analytically and concceptually tractable, and also helps us focus on what is important to the system. . If we want to model the pendulum, what should we choose to ignore? The light filtering through the window is obviously irrelevant, and if you take my word for it, air resistance is safely ignored. Let us also 



a spherical weight attached to a rigid rod of zero mass
But what matters is not the accuracy of our model but the significance of its terms. Yes, air exists and air resistance is troublesome, but it is more troublesome to a fighter jet than to a grandfather clock.

 (note that when it is clear from context that a system is a dynamical system, engineers often refer to it as a "system" to save precious time

 
The States of a Dynamical System
================================


In the same way a system is a collection of objects which we choose to consider, the ***states** of a system are the aspects of those objects we are interested in. Consider the pendulum. If we want to model the motion of a pendulum, what aspects of the system do we need to consider? Well, if by "motion of the pendulum" we mean  "the position and velocity of the weight along its semicircular trajectory at some time t given an initial position and velocity", then we need to represent the pendulum system as an ODE and see what we need to solve it (assuming an initial condition is given). The equation is

.. math::

   \ddot{\theta}(t) + \frac{g}{\ell} \sin\theta(t) = 0

where :math:`\theta(t)` is the angular displacement of the pendulum from the vertical, :math:`g` is the acceleration due to gravity, and :math:`\ell` is the length of the pendulum. These define the states of the system; given these values and an initial condition, we can solve the ODE and characterize the motion of the system over time. We may represent the states in column vector form, 



. What aspects of the system to we need to pay attention to  are likely interested in the  mechanical engineers, there are two : where , but have you considered that the rod might be gray while the ball is black, or that the rod and ball are both 60 C? In all likelihood, no. The pendulum is a dynamical system, and what we care about is the position and speed of the ball at any given time. Hence, the states of this system are its mass and the length of the rod; if we keep track of these, then given any initial condition, we can compute the positions and speed of the pendulum.

If we were biologists studying the camoflauge reflex of a startled squid, then the state of our squid-system would be the color of our squid. If we were chemical engineers studying the thermal reacticity of a chemical reagent, then the state of our system would be the temperature of the solution. 





The State-Space Representation of a Dynamical System
====================================================
The **state-space** representation of a dynamical system is the combination of two schools of thought. The first is interested in how a system evolves over time. This is primarily the realm of physicists and biologists and everything in between. The second is concerned with analyzing systems by its inputs and outputs. This is the ilk of the electrical engineer and his cousing, the computer engineer. When systems have too much complexity, we characterize its behaviour by what goes in and what comes out.


Formal Mathematical Definition
------------------------------
For a ***continuous-time time-varying system**, the standard state space model of a dynamical system is given by

.. math::

   \begin{aligned}
   \dot{x} &= f(x,\,u,\,t) + w(t) \\
   y &= h(x,\,u,\,t) + r(t) 
   \end{aligned}

where :math:`x \in \mathbb{R}^{n}` is the **state vector**, :math:`u \in \mathbb{R}^{p}` is the **input vector**, :math:`t \in \mathbb{R}^{+}` is **time**, and :math:`w` is a random variable referred to as **process noise** with :math:`w(t) \in \mathbb{R}^{n}` distributed as :math:`w \sim \mathcal{N}(0,\,Q(t))`, where :math:`Q(t) \in \mathcal{M}_{n}(\mathbb{R})` is the positive semidefinite **process noise covariance matrix**. Similarly, :math:`y \in \mathbb{R}^{m}` is the **measurement vector** and :math:`r` is a random variable referred to as **measurement noise** with  :math:`r(t) \in \mathbb{R}^{m}` distributed as :math:`r \sim \mathcal{N}(0,\,R(t))`, where :math:`R(t) \in \mathcal{M}_{m}(\mathbb{R})` is the positive semidefinite **measurement noise covariance matrix**.

The vector space \( X ⊆ \mathbb{R}^{n} \) is the **state-space**, the vector space \( Y \subseteq \mathbb{R}^{m} \) is the **measurement-space**, the vector space \( U \subseteq \mathbb{R}^{p} \) is the **input-space**, and the space \( T \subseteq \mathbb{R}^{+} \) is the time-space. The random variables are elements of the appropriate  Hilbert spaces. Recall that the derivitive is a linear operator \( \frac{d}{dt}: X \rightarrow X \); accordingly, the function \( f: X \times U \times T \rightarrow X\) is known as the **dynamics** function. Similarly, \( h:X \times U \times T \rightarrow Y \) is the **measurements ** function. 

For a **discrete-time time-varying system**, the standard state space model is given by

.. math::

   \begin{aligned}
     x_{k+1} &= f(x_k,\,u_k,\,t_k) + w_k \\
     y_k &= h(x_k,\,u_k,\,t_k) + r_k
   \end{aligned}

where :math:`x_k \in \mathbb{R}^{n}` is the state vector at time step :math:`k`, :math:`u_k \in \mathbb{R}^{p}` is the input vector, :math:`t_k \in \mathbb{R}^{+}` is the current time, and :math:`w_k \in \mathbb{R}^{n}` is a random variable distributed as :math:`w_k \sim \mathcal{N}(0,\,Q_k)` where :math:`Q_k \in \mathcal{M}_{n}(\mathbb{R})` is the positive semidefinite process noise covariance matrix.

Similarly, :math:`y_k \in \mathbb{R}^{m}` is the measurement vector and :math:`r_k \sim \mathcal{N}(0,\,R_k)`, where :math:`R_k \in \mathcal{M}_{m}(\mathbb{R})` is the positive semidefinite measurement noise covariance matrix. For a **discrete-time time-invariant system**, the functions :math:`f`, :math:`h`, :math:`w_k`, and :math:`r_k` are assumed to be independent of time.

The State Space Model in Python
-------------------------------
We may represent any of the systems above in the `pykal` library using the ``System`` class. This class is initialized with four functions: the state and measurement dynamics and the process and measurement noise functions, like so:

.. code-block:: python

   import numpy as np
   from pykal.system import System, SystemType
   from numpy.typing import NDArray

   def f(x: NDArray, u: NDArray, t: float) -> NDArray:
       return np.array([[x[1, 0]], [-x[0, 0]]])  # Harmonic oscillator

   def h(x: NDArray, u: NDArray, t: float) -> NDArray:
       return x[:1]  # Observe position

   def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
       return 0.01 * np.eye(2)

   def R(x: NDArray, u: NDArray, t: float) -> NDArray:
       return 0.1 * np.eye(1)

   def u(t: float) -> NDArray:
       return np.zeros((1, 1))

   sys = System(
       f=f,
       h=h,
       Q=Q,
       R=R,
       u=u,
       state_names=["x0", "x1"],
       measurement_names=["x0"],
       system_type=SystemType.CONTINUOUS_TIME_INVARIANT,
   )

The Limitations of Python
-------------------------

However, note that we find ourselves in a Pythonic pickle: while it is simple to initialize such a class with functions that are strictly analogous to their mathematical origin, what if our state dynamics only depended upon x, that is, ``f(x)`` instead of ``f(x,u,t)``? Any subclasses that utilize ``f`` should (rightly) expect it to have three arguments, so automated function calls become an issue.

Even if our hypothetical function is defined with all three inputs, what if the order is different, e.g. ``f(t, x, u)``? (For instance, SciPy expects its functions in ``f(t, x, *kwargs)`` format, while JAX prefers ``f(x, u, t)`` or ``f(x, t)`` for time-varying systems in its differential equation solvers). What if the function only accepts ```**kwargs`` like ``x``, ``t``, and ``u``? What if those keys are instead labeled ``state``, ``time``, and ``input``? What if one function expects an ``NDArray``, another expects a list, still another a dictionary, or no type annotations at all?

But now lets suppose that input wasn't an issue. What if a bug or a misimplementation causes the function to return an output with incorrect dimensions—for example, ``xdot.shape = (m_measurements, 1)`` instead of the expected ``xdot.shape = (n_states, 1)``, or perhaps  ``xdot.shape = (n_states,)``?

When building estimation pipelines, we often pass functions around as first-class objects:
dynamics models, measurement maps, Jacobians, noise models, and control inputs. In a well-structured system,
each of these should have a consistent interface—both in terms of the parameters they accept and the
outputs they return.

Unfortunately, Python’s flexibility can be a double-edged sword. While it enables rapid development,
it also allows functions with inconsistent signatures, missing type annotations, ambiguous return types,
or dynamic behavior that can silently break downstream logic.

Consider the contrast with C:

.. code-block:: c

   // C: fixed signature, explicit memory, compiler-enforced interface
   void dynamics(const double *x, const double *u, double t, double *x_dot_out) {
       x_dot_out[0] = x[1];
       x_dot_out[1] = -x[0];
   }

In C, functions must declare exactly what inputs they need and what outputs they produce.
Return types and pointer sizes are enforced at compile time, and passing the wrong number
of arguments or mismatched types raises immediate, traceable errors.

In Python, however, the same function might look like:

.. code-block:: python

   def f(x, u, t):
       return np.vstack([x[1], -x[0]])

But what if the caller forgets to provide `u`, or passes a scalar `t` as a list? Or worse,
what if the function silently returns a list instead of a NumPy array?

These kinds of mismatches are notoriously difficult to trace—especially when using external libraries
like TensorFlow, PyTorch, or JAX that wrap or recompile functions dynamically.

Without a consistent interface, chaos creeps in. Hence, a utility class was created to solve this problem
throughout the framework: ``SafeIO``.

This class validates user-defined functions at the time of registration (not just runtime),
injects only the necessary arguments, and enforces that all returned values are properly typed and shaped.
It brings the structure and safety of compiled languages to Python's dynamic, high-level ecosystem.

 
utils_safeio.py — Validated Function Dispatch and Signature Checking
=====================================================================

The module ``pykal.utils.utils_safeio`` provides centralized, decorator-based validation
for all vector-valued system functions used in simulation, estimation, and filtering.

It ensures that all user-defined functions (e.g., dynamics, measurement models, noise) conform
to expected signature conventions, use recognized variable names, and return well-typed outputs.

Function Signature Validation
-----------------------------

The decorator ``verify_signature_and_parameter_names(set_func: Callable) -> Callable``
is used throughout the framework to enforce the structure of user-defined functions.

This decorator checks:
- That only recognized parameter names are used (e.g., ``x``, ``u``, ``t``, and aliases).
- That each parameter is properly annotated with a type (e.g., ``NDArray``, ``float``).
- That the return type is an ``NDArray``.

Example usage — decorating a setter:

.. code-block:: python

   from pykal.utils.utils_safeio import SafeIO
   from numpy.typing import NDArray
   from typing import Callable, Optional

   class SystemIO:
       @SafeIO.verify_signature_and_parameter_names
       @staticmethod
       def set_f(f: Optional[Callable]) -> Callable:
           if f is None:
               return lambda t: np.zeros((1, 1))
           return f

   def good_f(x: NDArray, u: NDArray, t: float) -> NDArray:
       return x + u

   SystemIO.set_f(good_f)  # ✅ passes validation

   def bad_f(z: NDArray) -> NDArray:
       return z

   SystemIO.set_f(bad_f)  # ❌ raises TypeError: parameter 'z' is not a recognized alias

Calling Functions with Required Arguments
-----------------------------------------

Some user-defined functions only require a subset of inputs (e.g., ``x`` and ``t``, but not ``u``).
To support this flexibility, ``SafeIO`` provides a dispatch utility that selectively passes
only the arguments explicitly declared by the target function.

Example — using `smart_call`:

.. code-block:: python

   def f_partial(x: NDArray, t: float) -> NDArray:
       return x * t

   x = np.array([[1.0], [2.0]])
   t = 0.5

   y = SafeIO.smart_call(f_partial, x=x, t=t)  # ✅ passes x and t only

   def f_nothing() -> NDArray:
       return np.ones((2, 1))

   y2 = SafeIO.smart_call(f_nothing, x=x, t=t)  # ✅ ignores unused x and t

Full Input/Output Validation
----------------------------

The high-level utility ``call_validated_function_with_args(...)`` combines the signature
flexibility of ``smart_call`` with additional checks on output shape and type.

This ensures that even complex black-box models — such as trained neural networks — can
be integrated safely into the framework, so long as they conform to the expected interfaces.

Example — static validation with expected shape:

.. code-block:: python

   def fx(x: NDArray) -> NDArray:
       return x * 2

   x = np.array([[1.0], [2.0]])
   y = SafeIO.call_validated_function_with_args(fx, x=x, expected_shape=(2, 1))  # ✅

   def fx_wrong(x: NDArray) -> NDArray:
       return np.zeros((3, 1))

   y_bad = SafeIO.call_validated_function_with_args(fx_wrong, x=x, expected_shape=(2, 1))  # ❌ ValueError

Examples: Wrapping Machine Learning Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This utility makes it easy to wrap black-box inference models for use as measurement functions
or learned dynamics models.

**Keras model wrapper example:**

.. code-block:: python

   import numpy as np
   from tensorflow.keras.models import load_model

   keras_model = load_model("my_model.h5")

   def keras_wrapper(x: NDArray, t: float) -> NDArray:
       x_aug = np.vstack([x, [[t]]])  # time-augmented input
       return keras_model.predict(x_aug.T).T  # Keras expects (batch, features)

   x = np.array([[0.1], [0.2]])
   t = 0.5

   y = SafeIO.call_validated_function_with_args(
       keras_wrapper,
       x=x,
       t=t,
       expected_shape=(1, 1)
   )

**PyTorch model wrapper example:**

.. code-block:: python

   import torch
   import numpy as np

   class MyTorchModel(torch.nn.Module):
       def forward(self, x):
           return x.sum(dim=-1, keepdim=True)

   torch_model = MyTorchModel()

   def torch_wrapper(x: NDArray, u: NDArray) -> NDArray:
       x_u = np.vstack([x, u])
       x_tensor = torch.from_numpy(x_u.T).float()
       with torch.no_grad():
           out = torch_model(x_tensor).numpy().T  # shape: (1, n)
       return out

   x = np.array([[0.1], [0.2]])
   u = np.array([[0.3]])

   y = SafeIO.call_validated_function_with_args(
       torch_wrapper,
       x=x,
       u=u,
       expected_shape=(1, 1)
   )

These examples demonstrate how ``SafeIO`` enables compatibility with external machine learning libraries
while enforcing strong validation of runtime I/O behavior.

.. admonition:: Debugging Validation Errors in ``SafeIO``
   :class: toggle


   When using decorators or dispatch utilities from ``SafeIO``, validation errors are raised
   as soon as a user-defined function fails to meet expected interface rules. These errors
   are explicitly typed (``TypeError``, ``ValueError``) and include informative messages.

   Common causes and how to resolve them:

   **1. Unrecognized parameter name**

   .. code-block:: python

      def bad_f(z: NDArray) -> NDArray:
          return z

      SystemIO.set_f(bad_f)  # ❌

   Traceback:

   ::

      TypeError: In function `bad_f`, parameter `z` is not a recognized alias.
      Expected one of: ['input', 'state', 't', 't_k', 'tau', 'time', 'tk', 'u', 'u_k', 'uk', 'x', 'x_k', 'xk']

   **Fix**: Rename the parameter to a recognized alias like ``x`` or ``u``.

   ---

   **2. Missing type annotation**

   .. code-block:: python

      def bad_f(x) -> NDArray:
          return x

      SystemIO.set_f(bad_f)  # ❌

   Traceback:

   ::

      TypeError: In function `bad_f`, input `x` is missing a type annotation.

   **Fix**: Add a valid annotation, e.g. ``x: NDArray``.

   ---

   **3. Incorrect type for input**

   .. code-block:: python

      def bad_f(x: float) -> NDArray:
          return np.array([[x]])

      SystemIO.set_f(bad_f)  # ❌

   Traceback:

   ::

      TypeError: In function `bad_f`, input `x` must be NDArray[...] or Optional[NDArray], got <class 'float'>

   **Fix**: Change the annotation to ``x: NDArray``.

   ---

   **4. Invalid return type**

   .. code-block:: python

      def bad_f(x: NDArray) -> list:
          return [1, 2]

      SafeIO.call_validated_function_with_args(bad_f, x=np.ones((2, 1)))  # ❌

   Traceback:

   ::

      TypeError: Function must return a NumPy ndarray, got <class 'list'>

   **Fix**: Ensure the return value is a NumPy array and the return type is ``-> NDArray``.

   ---

   **5. Output shape mismatch**

   .. code-block:: python

      def bad_shape(x: NDArray) -> NDArray:
          return np.zeros((3, 1))

      SafeIO.call_validated_function_with_args(bad_shape, x=np.ones((2, 1)), expected_shape=(2, 1))  # ❌

   Traceback:

   ::

      ValueError: Output shape mismatch. Expected (2, 1), got (3, 1)

   **Fix**: Either fix the output shape or adjust the ``expected_shape=...`` argument.

   ---

   **Tips**

   - If you're not sure what aliases are allowed, check the sets:
     ``SafeIO._alias_for_x``, ``_alias_for_u``, and ``_alias_for_t``.
   - Use ``print(inspect.signature(func))`` to introspect user functions during debugging.
   - Wrap machine learning models with intermediate print/debug logic to inspect shapes before returning.

   This early validation design ensures that errors are caught before any downstream computation,
   making debugging localized, informative, and easy to isolate.


See Also
--------

- ``SystemIO``: Function registration and dispatch within the `System` class.
- ``utils_computation``: Derivative and Jacobian inference utilities.
- ``call_validated_function_with_args``: Core I/O wrapper used throughout simulation and estimation.


system.py — Instantiating and Simulating a Dynamical System
============================================================

The ``System`` class provides a validated interface for representing dynamical systems
with continuous- or discrete-time evolution, measurement functions, and optional noise models.
It accepts user-supplied functions for dynamics, measurement, input, and noise — all statically
validated and callable with flexible parameter sets (``x``, ``u``, ``t`` and their aliases).

Example: Creating a Continuous-Time System
------------------------------------------

The following example constructs a continuous-time harmonic oscillator:

.. code-block:: python

    import numpy as np
    from pykal.system import System, SystemType
    from numpy.typing import NDArray

    def f(x: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[x[1, 0]], [-x[0, 0]]])  # Harmonic oscillator

    def h(x: NDArray, u: NDArray, t: float) -> NDArray:
        return x[:1]  # Observe only position

    def Q(x: NDArray, u: NDArray, t: float) -> NDArray:
        return 0.01 * np.eye(2)

    def R(x: NDArray, u: NDArray, t: float) -> NDArray:
        return 0.1 * np.eye(1)

    def u(t: float) -> NDArray:
        return np.zeros((1, 1))  # Zero input

    sys = System(
        f=f,
        h=h,
        Q=Q,
        R=R,
        u=u,
        state_names=["x0", "x1"],
        measurement_names=["x0"],
        system_type=SystemType.CONTINUOUS_TIME_INVARIANT,
    )

Functional Interface, Not Data
------------------------------

Note that the ``System`` object contains **no state trajectory data**.
It simply wraps and validates user-defined functions. The functions are checked for:

- Correct parameter names (``x``, ``u``, ``t``) and aliases
- Proper type annotations (``NDArray``, ``float``, etc.)
- Matching return shapes for noise matrices and Jacobians

To generate data from a system, you must explicitly **simulate** its states and measurements.

Simulating System States
------------------------

Use the ``simulate_states()`` method to generate a state trajectory from an initial condition:

.. code-block:: python

    x0 = np.array([[1.0], [0.0]])
    X, T = sys.simulate_states(t_span=(0.0, 1.0), dt=0.1, x0=x0, process_noise=False)

This call:

- Integrates the system forward in time using ``f(x, u, t)``
- Uses the input function ``u(t)``
- Optionally injects process noise from ``Q(x, u, t)``
- Returns a state matrix ``X`` of shape ``(n_states, n_steps)`` and time vector ``T``

Simulating System Measurements
------------------------------

Use the ``simulate_measurements()`` method to evaluate the measurement function across a known state trajectory:

.. code-block:: python

    Y, T_meas = sys.simulate_measurements(X=X, T=T, measurement_noise=False)

This evaluates ``h(x, u, t)`` at each time step, optionally adding Gaussian noise from ``R(x, u, t)``.

Changing System Methods
-----------------------

System components like ``f``, ``h``, ``Q``, ``R``, and ``u`` can be reassigned after construction:

.. code-block:: python

    def new_f(x: NDArray, u: NDArray, t: float) -> NDArray:
        return x + u  # New dynamics

    sys.f = new_f

All reassigned functions are passed through the same static validation machinery as during instantiation.

Using ``override_*`` Keywords for Temporary Substitution
---------------------------------------------------------

For temporary substitution without permanently changing the system attributes, use the ``override_*`` keywords in simulation:

.. code-block:: python

    X_alt, T_alt = sys.simulate_states(
        x0=x0,
        t_span=(0.0, 1.0),
        dt=0.1,
        process_noise=False,
        override_system_f=new_f  # Use temporary dynamics
    )

Each ``override_*`` keyword follows the same pattern:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Keyword Value
     - Behavior
   * - ``False`` (default)
     - Use the method defined on ``sys``
   * - ``None``
     - Use a safe fallback (e.g. zero input) or raise if invalid
   * - ``Callable``
     - Use the provided function instead

This allows flexible experimentation and testing with alternate models, without modifying the system object.

_base_kf.py — Kalman Filter Base Classes
==========================

The `pykal` Kalman filter framework is built upon a small hierarchy of validated abstract base classes. These include:

- `BaseKF` — the main interface for defining predict-update-run workflows
- `BaseKFSqrt` — a square-root variant that uses Cholesky and QR for numerical stability
- `BaseKFPartialUpdate` — an extension that allows per-state partial updates using a time-varying β matrix

Each subclass enforces a consistent filtering contract while enabling flexibility in implementation.

BaseKF — Core Kalman Filter Interface
-------------------------------------

The `BaseKF` class defines the high-level structure and required methods for all Kalman filters. It is not meant to be used directly, but rather extended by specific implementations like the EKF or UKF.

Initialization
^^^^^^^^^^^^^^

To use a `BaseKF` subclass, you must pass in a validated `System` object:

.. code-block:: python

   from pykal.ekf import EKF  # subclass of BaseKF
   from pykal.system import System

   ekf = EKF(sys)

Required Methods
^^^^^^^^^^^^^^^^

All subclasses must implement the following:

- ``_predict(...)``: Propagates the mean and covariance.
- ``_update(...)``: Corrects the estimate using new measurements.
- ``run(...)``: Executes the full filtering loop.

.. code-block:: python

   # All filters must implement this API
   x_pred, P_pred = self._predict(xk, Pk, Fk, Qk, dt, uk, tk, square_root)
   x_upd,  P_upd  = self._update(xk, Pk, yk, y_pred, Hk, Rk, dt, uk, tk, square_root, beta_mat)
   X, P, T        = self.run(x0=x0, P0=P0, Y=Y, start_time=0.0, dt=0.1)

BaseKFSqrt — Square-Root Kalman Filter
--------------------------------------

The `BaseKFSqrt` subclass provides numerically stable versions of prediction and update steps using QR and Cholesky decompositions.

.. code-block:: python

   from pykal.ekf import EKF

   ekf_sqrt = EKF(sys)
   X, P, T = ekf_sqrt.run(
       x0=x0,
       P0=P0,
       Y=Y,
       dt=0.1,
       start_time=0.0,
       square_root=True  # Enable SRKF
   )

Under the hood, square-root propagation is performed via:

.. code-block:: python

   # Prediction
   FS = Fk @ chol(Pk)
   Q_sqrt = chol(Qk)
   A = np.hstack((FS, Q_sqrt))
   _, R = np.linalg.qr(A.T)
   P_pred = R.T @ R

   # Update
   HS = Hk @ chol(Pk)
   R_sqrt = chol(Rk)
   A = np.vstack((HS, R_sqrt))
   _, R = np.linalg.qr(A)
   P_upd = R.T @ R

BaseKFPartialUpdate — Per-State β-Weighting
-------------------------------------------

The `BaseKFPartialUpdate` class allows the user to control how much each state is corrected by defining a β(t) matrix with entries in [0, 1].

Setting β(t)
^^^^^^^^^^^^

You may define a β function manually:

.. code-block:: python

   def beta_fn(t: float) -> NDArray:
       return np.array([[1.0], [0.0], [0.5]])  # update only state 0 fully, 2 partially

   pskf = PSKF(sys, beta=beta_fn)

Or let it default to full update:

.. code-block:: python

   pskf = PSKF(sys)  # β(t) ≡ ones(n, 1)

Update Equation
^^^^^^^^^^^^^^^

The covariance update becomes:

.. math::

   P_{\text{upd}} = \beta P_{\text{base}} \beta^\top + (I - \beta) P_k (I - \beta)^\top

This allows the correction to be selectively suppressed for individual states.

Debugging Filter Errors
-----------------------

.. admonition:: Expand for debugging tips
   :class: dropdown

   Common validation errors include:

   - **Function signature mismatch**: All user-supplied functions must use valid aliases (e.g., `x`, `u`, `t`) and type annotations (`NDArray` or `float`).
   - **Missing return type**: `f`, `h`, `Q`, and `R` must return `NDArray` and declare this explicitly.
   - **Shape mismatch**: Ensure `Q` and `R` return matrices of shape `(n, n)` and `(m, m)` respectively, matching the number of states and measurements.

   To debug:

   .. code-block:: python

      from pykal.utils.utils_safeio import SafeIO
      SafeIO.call_validated_function_with_args(f, x=x, u=u, t=t, expected_shape=(n, 1))

   This helps isolate where your function fails the validation pipeline.

Inheritance Diagram
-------------------

.. graphviz::

   digraph FilterInheritance {
       rankdir=LR;
       node [shape=box, style="filled", fillcolor="#f0f0f0"];

       BaseKF -> BaseKFSqrt;
       BaseKF -> BaseKFPartialUpdate;

       BaseKF [label="BaseKF (Abstract)", fillcolor="#d0e0ff"];
       BaseKFSqrt [label="BaseKFSqrt"];
       BaseKFPartialUpdate [label="BaseKFPartialUpdate"];
   }


ekf.py — Extended Kalman Filter with Optional Partial Updates
=============================================================

The :class:`~pykal.ekf.EKF` class implements the Extended Kalman Filter (EKF) with support for:

- **Partial updates** using a time-varying state-weighting function :math:`\\beta_k`
- **Square-root filtering** for numerically stable covariance propagation

This unified filter class generalizes multiple variants:

- Standard EKF (β = 1)
- Schmidt-Kalman Filter (SKF) (some βᵢ = 0)
- Partial EKF (PEKF) (0 < βᵢ < 1)
- Square-root EKF (SR-EKF)
- Partial square-root EKF (PSR-EKF)

Input validation is enforced through :class:`~pykal.ekf.EKFIO`, which checks initial conditions, measurement dimensions, runtime override options, and callable types.

Motivation
----------

The base EKF updates all states uniformly. However, when confidence or observability is state-dependent, a partial update matrix βₖ allows each state to be updated proportionally. This is useful when:

- Some states are weakly observable or unobservable
- You want to freeze uncertain state components (e.g., in SKF)
- Numerical instability arises in full covariance updates

The EKF class unifies these features while supporting both standard and square-root update forms.

Quick Example
-------------

.. code-block:: python

    >>> import numpy as np
    >>> from numpy.typing import NDArray
    >>> from pykal.system import System, SystemType
    >>> from pykal.ekf import EKF

    # Define system dynamics and Jacobians
    >>> def f(x, u, t): return np.array([[x[1, 0]], [-x[0, 0]]])
    >>> def h(x, u, t): return x[:1]
    >>> def F(x, u, t): return np.array([[0., 1.], [-1., 0.]])
    >>> def H(x, u, t): return np.array([[1.0, 0.0]])
    >>> def Q(x, u, t): return 0.01 * np.eye(2)
    >>> def R(x, u, t): return 0.1 * np.eye(1)
    >>> def u(t): return np.zeros((1, 1))

    >>> sys = System(
    ...     f=f, h=h, Q=Q, R=R, u=u, F=F, H=H,
    ...     state_names=["x0", "x1"],
    ...     measurement_names=["x0"],
    ...     system_type=SystemType.DISCRETE_TIME_INVARIANT,
    ... )

    >>> x0 = np.array([[0.0], [1.0]])
    >>> P0 = 0.01 * np.eye(2)

    >>> X_ref, T = sys.simulate_states(x0=x0, dt=0.1, t_span=(0, 1), process_noise=False)
    >>> Y, _ = sys.simulate_measurements(X=X_ref, T=T, measurement_noise=False)

    >>> ekf = EKF(sys)

    >>> # === Run full EKF (β = 1) ===
    >>> X_est, P_est, T_out = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     start_time=0.0, dt=0.1, square_root=False,
    ...     override_system_beta=False)

    >>> X_est.shape == (Y.shape[1] + 1, 2)  # +1 for initial state
    True

    >>> np.allclose(X_est[0], x0.flatten())  # initial state correct
    True
    >>> not np.allclose(X_est[1], x0.flatten())  # state has evolved
    True

    >>> # === Square-root EKF ===
    >>> X_sqrt, P_sqrt, _ = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     start_time=0.0, dt=0.1, square_root=True,
    ...     override_system_beta=False)

    >>> np.allclose(X_est, X_sqrt, atol=1e-5)  # state trajectories agree
    True
    >>> np.allclose(P_est[-1], P_sqrt[-1], atol=1, rtol=1)  # final covariances match
    True

    >>> # === Freeze second state (β = [1, 0]) ===
    >>> ekf.beta = lambda t: np.array([[1.0], [0.0]])
    >>> X_partial, _, _ = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     start_time=0.0, dt=0.1,
    ...     square_root=False, override_system_beta=False)

    >>> np.allclose(X_partial[:, 1], x0[1, 0])  # second state frozen
    True

    >>> # === No updates at all (β = 0) ===
    >>> X_frozen, _, _ = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     start_time=0.0, dt=0.1,
    ...     square_root=False, override_system_beta=None)

    >>> np.allclose(X_frozen, np.tile(x0.flatten(), (Y.shape[1] + 1, 1)))
    True

    >>> # === Check uncertainty evolution ===
    >>> def beta_fn(t): return np.array([[1.0], [0.0]])
    >>> ekf.beta = beta_fn
    >>> _, P_skf, _ = ekf.run(
    ...     x0=x0, P0=P0, Y=Y.T,
    ...     start_time=0.0, dt=0.1, override_system_beta=False)

    >>> all(np.isclose(P[1, 1], P0[1, 1]) for P in P_skf)  # second state's uncertainty constant
    True
    >>> any(P[0, 0] < P0[0, 0] for P in P_skf)  # first state's uncertainty reduces
    True

Design Notes
------------

- The filter relies on `call_validated_function_with_args` for every Jacobian and noise function call to ensure type- and shape-safety.
- Time-varying β(t) must be a callable returning a column vector of shape (n, 1)
- If `square_root=True`, QR-based updates are used for numerical stability
- For reproducibility, use `square_root=False` unless the problem is ill-conditioned

Related Classes
---------------

- :class:`~pykal._base_kf.BaseKFPartialUpdate` — adds β-matrix handling
- :class:`~pykal._base_kf.BaseKFSqrt` — adds QR-based square-root logic
- :class:`~pykal.ekf.EKFIO` — input validation for EKF configuration and overrides
