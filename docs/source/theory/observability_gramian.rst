Observability Gramian
=====================

The **observability Gramian** provides a quantitative test of whether a system’s internal state can be inferred from its outputs over a finite time window.

Continuous-Time Linear Systems
------------------------------

Consider the linear time-invariant (LTI) system:

.. math::

   \dot{x}(t) = A x(t), \qquad y(t) = C x(t),

where:

- :math:`x(t) \in \mathbb{R}^n`: state vector
- :math:`y(t) \in \mathbb{R}^m`: output vector
- :math:`A \in \mathbb{R}^{n \times n}`: system matrix
- :math:`C \in \mathbb{R}^{m \times n}`: output matrix

The **observability Gramian** over the interval :math:`[t_0, t_1]` is defined as:

.. math::

   W_o(t_0, t_1) = \int_{t_0}^{t_1} \Phi(t, t_0)^\top\, C^\top C\, \Phi(t, t_0)\,\mathrm{d}t,

where :math:`\Phi(t, t_0) = e^{A(t - t_0)}` is the state transition matrix.

Interpretation:

- If :math:`W_o` is **positive definite**, the system is **observable** over :math:`[t_0, t_1]`.
- If :math:`W_o` is **singular**, some directions in state space are **unobservable**.

Discrete-Time Case
------------------

For the discrete-time system:

.. math::

   x_{k+1} = A x_k, \qquad y_k = C x_k,

the observability Gramian over :math:`k = 0, \dots, N-1` is:

.. math::

   W_o(0, N) = \sum_{k=0}^{N-1} (A^k)^\top C^\top C A^k

This matrix plays the same role: it tells whether internal state directions affect the output over the window.

Rank and Observability
----------------------

- :math:`\mathrm{rank}(W_o) = n` ⇔ full observability
- :math:`\mathrm{rank}(W_o) < n` ⇔ partial observability

If :math:`W_o` is full rank, one can recover the initial state :math:`x(0)` (or parameter vector in estimation contexts) from measurements over the interval.

Least Squares Connection
------------------------

In batch estimation of :math:`x` from data :math:`y_k = C A^k x`, the least-squares cost becomes:

.. math::

   \sum_k \lVert C A^k x - y_k \rVert_2^2
   = x^\top W_o x - 2 x^\top b + \text{const.},

where :math:`W_o` appears as the quadratic term. The normal equations become:

.. math::

   W_o x^* = b.

Hence, **invertibility of the Gramian = unique least-squares solution**.

Sensor Design Insight
---------------------

The matrix :math:`C` determines how internal states affect outputs. If some rows of :math:`C` are zero or linearly dependent, certain states may be unmeasured or indistinguishable.

- To **maximize observability**, choose sensors (rows of :math:`C`) that excite distinct state directions.
- Observability analysis helps in **sensor placement**, **model reduction**, and **filter tuning**.

Practical Applications
----------------------

- In Kalman filters: observability determines whether the filter can track all states.
- In system identification: poor observability leads to unidentifiable models.
- In control design: feedback must act on observable states.

Visualization Tip
-----------------

Plotting eigenvalues of :math:`W_o` over time (e.g., in a sliding window) helps detect observability loss in time-varying or nonlinear systems. Small eigenvalues imply weak observability.

Next: We interpret projections and residuals geometrically, linking them to null spaces and observability.

