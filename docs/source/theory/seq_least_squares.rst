Geometric and Observability Insights: Sequential Least Squares
===============================================================

Recursive Least Squares (RLS) is the sequential counterpart to batch least-squares. It processes one measurement at a time, updating the estimate and uncertainty incrementally. The same geometric and observability principles apply — but unfold over time.

Setup
-----

Let each observation at time step :math:`k` be:

- a row vector :math:`a_k^T \in \mathbb{R}^{1 \times n}` (regression input),
- a scalar measurement :math:`y_k \in \mathbb{R}`,
- a parameter estimate :math:`x_k \in \mathbb{R}^n`,
- a covariance matrix :math:`P_k \in \mathbb{R}^{n \times n}` (confidence in :math:`x_k`).

Initialization:

.. math::

   x_0 = \text{prior guess}, \qquad P_0 = \alpha\,I \quad (\alpha \text{ large}).

At each time step :math:`k`, do:

1. **Compute Kalman Gain**

   .. math::

      K_k = \frac{P_{k-1}\,a_k}{a_k^T\,P_{k-1}\,a_k + \sigma^2}

2. **Update Estimate**

   .. math::

      x_k = x_{k-1} + K_k\,\bigl(y_k - a_k^T x_{k-1}\bigr)

3. **Update Covariance**

   .. math::

      P_k = (I - K_k\,a_k^T)\,P_{k-1}

This is equivalent to batch least-squares after :math:`N` steps — but is far more efficient.

Geometric Interpretation
------------------------

Each observation :math:`(a_k^T, y_k)` defines a hyperplane in :math:`\mathbb{R}^n`:

.. math::

   a_k^T x = y_k

The update adjusts the prior estimate :math:`x_{k-1}` by **projecting the residual**:

.. math::

   r_k = y_k - a_k^T x_{k-1}

onto the vector :math:`P_{k-1} a_k`. That is:

.. math::

   x_k = x_{k-1} + (\text{scaled projection of } r_k)

The covariance update reflects the reduction in uncertainty along the direction of :math:`a_k`.

Observability in Sequential Estimation
--------------------------------------

In batch least-squares, observability is determined by the Gramian:

.. math::

   W = \sum_{k=1}^{N} a_k a_k^T = A^\top A.

In sequential estimation, the accumulated **information matrix** plays the same role:

.. math::

   M_k = \sum_{i=1}^{k} \frac{a_i a_i^T}{\sigma^2}

This matrix grows over time as more directions are measured. If the :math:`a_k` span :math:`\mathbb{R}^n`, then:

- :math:`M_k` becomes full rank
- The estimate :math:`x_k` converges
- Uncertainty :math:`P_k` shrinks in all directions

If the :math:`a_k` repeat or lie in a subspace, then some directions are never observed. These directions remain poorly estimated.

Key idea:  
**Observability ⇔ Persistent Excitation**

If the inputs do not excite all dimensions of the parameter space, observability is lost.

Practical Example
-----------------

Suppose :math:`x = (b_0, b_1)^T` is the offset and slope of a temperature sensor. The regressors are:

.. math::

   a_k^T = \begin{pmatrix}1 & T_k\end{pmatrix}

If all :math:`T_k` are equal, the second column of :math:`A` is constant ⇒ rank-deficient ⇒ :math:`b_1` is unobservable.

Sequentially:

- :math:`P_k` shrinks only in the direction of the constant regressors
- :math:`b_1` remains uncertain indefinitely

Interpretation
--------------

- The covariance matrix :math:`P_k` reflects **how observable** each direction is
  - Small eigenvalue ⇒ high confidence
  - Large eigenvalue ⇒ low confidence
- Observability is not static — it evolves as data arrive
- Plotting the eigenvalues or condition number of :math:`P_k` over time reveals which parameters are becoming observable

Connection to Filtering
-----------------------

This RLS scheme is the **Kalman filter** for:

- static system dynamics (no state evolution),
- known measurement model :math:`y_k = a_k^T x + \varepsilon_k`,
- white Gaussian noise :math:`\varepsilon_k \sim \mathcal{N}(0, \sigma^2)`.

Hence, the Kalman filter **generalizes** RLS to dynamic models with process noise and nonlinearities.

Conclusion
----------

Sequential least-squares is more than an efficient algorithm — it is a window into how information accumulates.

- Each step projects onto a new data constraint
- The observability of parameters grows with data diversity
- RLS tracks both the estimate and our **confidence** in each direction

This dual interpretation — geometric and statistical — makes it ideal for online calibration and estimation.

Next: The Kalman Filter — optimal sequential estimation under uncertainty.

