Geometric Interpretation: Projections, Residuals, and Null Spaces
==================================================================

Least-squares estimation can be viewed through the lens of geometry — as a problem of **projections** in vector spaces. This interpretation yields deep insight into residuals, observability, and estimator structure.

Orthogonal Projections
----------------------

Given a linear model:

.. math::

   y = A x + \varepsilon,

we seek the best approximation :math:`x^*` such that :math:`A x^*` is close to :math:`y` in Euclidean norm. The **least-squares solution** satisfies:

.. math::

   A^\top (A x^* - y) = 0
   \quad\Rightarrow\quad
   x^* = (A^\top A)^{-1} A^\top y.

Define:

- :math:`r = y - A x^*`: the **residual**
- :math:`P = A (A^\top A)^{-1} A^\top`: the **projection matrix** onto :math:`\mathrm{Col}(A)`

Then:

.. math::

   A x^* = P y, \qquad r = (I - P) y.

That is:

- The **estimate** is the projection of the data onto the column space of :math:`A`
- The **residual** is orthogonal to the column space: :math:`A^\top r = 0`

Null Space and Non-Uniqueness
-----------------------------

If :math:`A` does not have full column rank, then:

- :math:`A^\top A` is singular
- The solution :math:`x^*` is **not unique**
- The general solution is:

  .. math::

     x^* + v, \qquad \text{for any } v \in \ker(A)

That is, we can add any vector from the null space of :math:`A` without changing the fit: :math:`A(x^* + v) = A x^*`.

Geometric View of Observability
-------------------------------

A state (or parameter) direction is **observable** if it affects the output — i.e., if it lies **outside** the null space of :math:`A`.

- If :math:`v \in \ker(A)`, then :math:`A v = 0`: this direction is **unobservable**
- If :math:`v \notin \ker(A)`, then :math:`A v \ne 0`: this direction **can be inferred**

Hence:

- Observability corresponds to **non-orthogonality** between state directions and the row space of :math:`A`
- The **Gramian** :math:`W = A^\top A` quantifies how observable each direction is
  - Small eigenvalue ⇔ weak observability
  - Zero eigenvalue ⇔ unobservable

Example: Sensor Calibration Revisited
-------------------------------------

Suppose we are calibrating a temperature sensor via

.. math::

   y = b_0 + b_1 T + \varepsilon.

Let:

.. math::

   A = \begin{pmatrix}
     1 & T_1 \\
     1 & T_2 \\
     \vdots & \vdots \\
     1 & T_N
   \end{pmatrix}, \qquad
   x = \begin{pmatrix} b_0 \\ b_1 \end{pmatrix}

Then:

- If all :math:`T_k` are the same, the second column is constant ⇒ rank-deficient ⇒ :math:`b_1` is **unobservable**
- If :math:`T_k` vary, :math:`A` has full column rank ⇒ both :math:`b_0` and :math:`b_1` are observable

Residual Orthogonality
-----------------------

The least-squares residual vector is always **orthogonal to the model space**:

.. math::

   A^\top (y - A x^*) = 0.

This is the first-order optimality condition: we have projected :math:`y` onto the affine subspace spanned by the columns of :math:`A`.

Observability and Projections
-----------------------------

Let’s link projections, residuals, and observability:

- **Observable directions** project non-trivially onto the column space of :math:`A`
- **Unobservable directions** lie in the null space and cannot be distinguished from zero
- The **residual** measures the discrepancy between the data and the model output

Thus:

.. math::

   \text{observability} \;\Leftrightarrow\; \text{invertibility of projection onto model space}.

Practical Implications
----------------------

- Rank-deficient data ⇒ ambiguous estimates ⇒ poor filter performance
- Visualization: plot the **singular values** of :math:`A` or :math:`W = A^\top A` to assess observability
- Covariance estimates in Kalman filters or RLS shrink in observable directions and stay large in unobservable ones

Next: We complete the picture with the Kalman filter, combining dynamic models with sequential least-squares in the presence of noise.

