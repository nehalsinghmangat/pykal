What is a Dynamical System?
===========================
.. epigraph::

   "All the world's a [dynamical system],
   
   And all the men and women merely [dynamical systems]."

   -- William Shakespeare (paraphrased)

Everything. Every drop of rain or ripple in a pond; the worms in the dirt and the wires in your brain. You drive a dynamical system to work; you watch a dynamical system rise in the morning and set in the evening; and don't look now, but a pair of dynamical systems are flicking across the screen as you read this text.

Simpy put, a **dynamical system** is a system that changes over time. Since this definition can conceivably apply to anything, it depends on our discretion to use it meaningfully. A gnat is a dynamical system of cells. A clock is a dynamical system of springs and gears.  We may concede that a broken clock or a dead gnat are systems, but they are dynamical no longer.

We often consider sufficiently complex dynamical systems to be composed of **subsystems** (e.g. every cell of a gnat is a world unto itself); conversly, we may claim our system itself is a subsystem of a larger **ecosystem** (consider how the average swamp is a buzzing of many gnats along with mosquitoes, flies, moths, and, why not, clocks.)

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
For a **continuous-time time-varying system**, the standard state space model of a dynamical system is given by

.. math::

   \begin{aligned}
   \dot{x} &= f(x,\,u,\,t) + w(t) \\
   y &= h(x,\,u,\,t) + r(t) 
   \end{aligned}

where :math:`x \in \mathbb{R}^{n}` is the **state vector**, :math:`u \in \mathbb{R}^{p}` is the **input vector**, :math:`t \in \mathbb{R}^{+}` is **time**, and :math:`w` is a random variable referred to as **process noise** with :math:`w(t) \in \mathbb{R}^{n}` distributed as :math:`w \sim \mathcal{N}(0,\,Q(t))`, where :math:`Q(t) \in \mathcal{M}_{n}(\mathbb{R})` is the positive semidefinite **process noise covariance matrix**. Similarly, :math:`y \in \mathbb{R}^{m}` is the **measurement vector** and :math:`r` is a random variable referred to as **measurement noise** with  :math:`r(t) \in \mathbb{R}^{m}` distributed as :math:`r \sim \mathcal{N}(0,\,R(t))`, where :math:`R(t) \in \mathcal{M}_{m}(\mathbb{R})` is the positive semidefinite **measurement noise covariance matrix**.

The vector space \( X ⊆ \mathbb{R}^{n} \) is the **state-space**, the vector space \( Y \subseteq \mathbb{R}^{m} \) is the **measurement-space**, the vector space \( U \subseteq \mathbb{R}^{p} \) is the **input-space**, and the space \( T \subseteq \mathbb{R}^{+} \) is the time-space. The random variables are elements of the appropriate  Hilbert spaces. Recall that the derivitive is a linear operator \( \frac{d}{dt}: X \rightarrow X \); accordingly, the function \( f: X \times U \times T \rightarrow X\) is known as the **dynamics** function. Similarly, \( h:X \times U \times T \rightarrow Y \) is the **measurements** function. 

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

