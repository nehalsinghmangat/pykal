Linear Least Squares and Orthogonal Projections
===============================================

Suppose the function :math:`f` is linear:  
:math:`f(x) = A x` for some known matrix :math:`A \in \mathbb{R}^{m \times n}`.

Then the least-squares problem becomes:

.. math::
   :label: eq:linear-lsq

   x^* = \arg\min_x \lVert A x - y \rVert_2^2

and the gradient condition simplifies to:

.. math::

   \nabla\phi(x) = A^\top(A x - y) = 0
   \quad\Longrightarrow\quad
   A^\top A x^* = A^\top y

These are the **normal equations**. Provided :math:`A^\top A` is invertible, the solution is:

.. math::
   x^* = (A^\top A)^{-1} A^\top y

This is known as the **Moore–Penrose pseudoinverse** solution:

.. math::
   x^* = A^{\dagger} y

Geometric Interpretation
------------------------

The equation :math:`Ax = y` may not have a solution if :math:`y \notin \mathrm{Col}(A)`. But the least-squares solution satisfies:

.. math::
   A x^* = \operatorname{proj}_{\mathrm{Col}(A)}(y)

That is, :math:`A x^*` is the **orthogonal projection** of :math:`y` onto the column space of :math:`A`:

.. math::

   A x^* = P_A y,
   \quad
   P_A = A (A^\top A)^{-1} A^\top

The projection matrix :math:`P_A` is symmetric and idempotent:

- :math:`P_A^2 = P_A`
- :math:`P_A^\top = P_A`

and satisfies :math:`P_A y = y` iff :math:`y \in \mathrm{Col}(A)`.

Hence, least-squares estimation geometrically finds the point in the model space closest to the observation vector.

Observability Viewpoint
------------------------

In this context, **observability** means that :math:`x^*` is **uniquely determined** by :math:`y`. This is the case if and only if:

.. math::
   \mathrm{rank}(A) = n

Define the **observability Grammian**:

.. math::

   W := A^\top A

Then:

- If :math:`W` is invertible: full observability, unique solution
- If :math:`W` is singular: unobservable directions exist in :math:`x`

This aligns with control-theoretic definitions of observability.

Example: Calibration with Linear Model
--------------------------------------

Suppose a voltage :math:`y` is related to temperature :math:`T` via:

.. math::

   y = b_0 + b_1 T + \varepsilon

We record :math:`N` calibration points:

.. math::

   A =
   \begin{pmatrix}
   1 & T_1 \\
   1 & T_2 \\
   \vdots & \vdots \\
   1 & T_N
   \end{pmatrix},
   \quad
   x = \begin{pmatrix} b_0 \\ b_1 \end{pmatrix},
   \quad
   y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{pmatrix}

Then the least-squares estimate is:

.. math::

   x^* = (A^\top A)^{-1} A^\top y

This gives a fitted model:

.. math::

   \hat y(T) = b_0^* + b_1^* T

Observability Insight:

- If all :math:`T_i` are equal: rank-deficient → :math:`b_1` unobservable
- If :math:`T_i` are spread: full-rank → both :math:`b_0` and :math:`b_1` observable

Hence, sensor calibration requires **input excitation** to estimate parameters uniquely.

Next: We extend this to polynomial sensor models using Vandermonde matrices.

