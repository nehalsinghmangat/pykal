Inverse Problems and Least-Squares
==================================

Let :math:`X \subseteq \mathbb{R}^{n}` and :math:`Y \subseteq \mathbb{R}^{m}` be Euclidean spaces of dimensions :math:`n` and :math:`m`, respectively. Let :math:`f\colon X \to Y` be a smooth function, and suppose we are given some :math:`y \in Y`. We wish to solve:

.. math::
   :label: eq:funeq

   f(x) = y

We say this equation has a **solution** if there exists an :math:`x \in X` such that :math:`f(x) = y`. Three canonical cases arise:

- **Injective.** If :math:`f` is one-to-one and :math:`y \in f(X)`, there is a **unique**
  solution :math:`x = f^{-1}(y)`. If :math:`y \notin f(X)`, **no** solution exists.

- **Surjective.** If :math:`f` is onto, then for **every** :math:`y \in Y` there is at
  least one :math:`x \in X` with :math:`f(x) = y` (but it may not be unique).

- **Bijective.** If :math:`f` is both injective and surjective, then there is a **unique**
  solution :math:`x = f^{-1}(y)` for **every** :math:`y \in Y`.

Thus, for all :math:`y \in Y` a solution exists for Eq.~:eq:`eq:funeq` except in the case when :math:`f` is injective.

“So what?”, you might think. “Having solutions in two out of three cases doesn’t seem so bad; and if we restrict our attention to only :math:`y \in f(X)`, then we have solutions in all cases.” 

But alas! A variety of sensors and physical systems are injective but not surjective—meaning solutions exist only when the data lie within the image of :math:`f` (and we are rarely so fortunate)

Motivating Example: When Inversion Fails
----------------------------------------

Consider an RTD (Resistance Temperature Detector), which converts temperature to resistance via:

.. math::

   f(T) = R_0 (1 + \alpha T), \quad T \in [T_{\min}, T_{\max}]

This function is injective, but not surjective—resistances outside the calibration range have no corresponding temperature.  Presented with such scenarios, should we give up on finding a solution to  Eq.~:eq:`eq:funeq` ? 

Yes. Let us instead define a new problem.

Minimize the Residual
---------------------

As before, let :math:`f\colon X \to Y` be a smooth function, and suppose we are given :math:`y \in Y`. We define the **residual** as

.. math::
   :label: eq:residual

   r(x) \;=\; f(x) - y

and the **2-norm** of any vector :math:`v` as

.. math::
   :label: eq:l2norm

   \lVert v\rVert_2 \;=\; \sqrt{v^\top v}\,.

Our new problem is this: we wish to minimize the residiual. That is, we wish to solve for an :math:`x \in X` such that the 2-norm of the residual is minimized (We call this the **least squares problem**.) More formally, we have

.. math::
   :label: eq:lsq_problem

   \hat{x} \;=\;\arg\min_{x \in X} \lVert f(x) - y\rVert_2^2\,.

What does that mean? It means that such an :math:`\hat{x}` may not be an exact solution to Eq.~:eq:`eq:funeq`, but it is the closest we can get (quite literally, as the 2-norm is the standard Euclidean norm and is how we define distance). What’s more, if a solution to Eq.~:eq:`eq:funeq` does exist, it will also be the solution to the least-squares problem.

Solving the Least-Squares Problem
---------------------------------

Define a **cost** function: a scalar quantity measuring the discrepancy between the prediction :math:`f(x)` and the observation :math:`y`. A natural choice is the squared 2-norm of the residual:

.. math::
   :label: eq:phi

   \phi(x) \;=\; \tfrac12\,\lVert f(x) - y\rVert_2^2.

We include the square and the 1/2 factor for the following reasons:

- **Squaring the 2-norm** ensures the cost is always non-negative and differentiable everywhere (unlike the absolute value norm).  
- It simplifies the derivative: the gradient of a squared norm gives a clean expression involving the Jacobian.
- The factor of 1/2 cancels the 2 in the derivative, making formulas neater:
  
  .. math::

     \nabla\phi(x) \;=\; J(x)^\top(f(x)-y).

- Geometrically, the squared 2-norm corresponds to **Euclidean distance squared**, which appears naturally in projection and optimization problems.
- It is the **maximum-likelihood cost** when the measurement noise is Gaussian: if :math:`y = f(x) + \varepsilon` with :math:`\varepsilon \sim \mathcal{N}(0, \sigma^2 I)`, then minimizing :math:`\lVert f(x)-y \rVert_2^2` is equivalent to maximizing the likelihood.

Thus, this cost function thus has theoretical, geometric, and statistical justification.

### 1. Calculus (First‐Order Condition)

To minimize the cost function

.. math::

   \phi(x) \;=\; \tfrac12\,\lVert f(x) - y\rVert_2^2,

we set its gradient to zero. This is the **first-order necessary condition** for a minimum.

Using the chain rule, the gradient is:

.. math::
   :label: eq:gradient

   \nabla\phi(x) \;=\; J(x)^{T}\bigl(f(x)-y\bigr),

where :math:`J(x) = \frac{\partial f(x)}{\partial x}` is the **Jacobian matrix** of the function :math:`f`.

This equation generalizes the familiar normal equations from linear least-squares. Solving for :math:`x^*` such that

.. math::

   J(x^*)^{T}\bigl(f(x^*) - y\bigr) = 0

yields a **stationary point** of the cost function. If :math:`f` is linear, the problem is convex and this point is the unique global minimizer. If :math:`f` is nonlinear, the problem may be non-convex and admit multiple solutions or local minima.

In such nonlinear cases, we typically solve this equation using **iterative methods**, such as:

- **Gradient Descent**  
  Update: :math:`x^{(k+1)} = x^{(k)} - \eta\,\nabla\phi(x^{(k)})`

- **Gauss–Newton Method**  
  Approximate the Hessian as :math:`J^T J`, solve a linear system at each iteration.

- **Levenberg–Marquardt Algorithm**  
  Interpolates between Gauss–Newton and gradient descent for improved robustness.

Each method attempts to drive the residual :math:`f(x)-y` toward zero by following the shape of the cost surface defined by :math:`\phi(x)`.

This calculus-based approach is general and forms the basis for solving both linear and nonlinear least-squares problems.

### 2. Orthogonal Projection (Linear Case)

If the function :math:`f` is linear, that is, :math:`f(x) = A x` for some matrix :math:`A \in \mathbb{R}^{m \times n}`, then its Jacobian is constant: :math:`J(x) = A`. Substituting into the gradient condition from Eq.~:eq:`eq:gradient` yields

.. math::

   A^{T}\bigl(Ax^* - y\bigr) = 0
   \quad\Longrightarrow\quad
   A^{T}A\,x^* = A^{T}y
   \quad\text{(normal equations).}

This system is solvable when :math:`A^T A` is invertible—i.e., when the columns of :math:`A` are linearly independent (full rank).

Solving for :math:`x^*` gives the unique least-squares solution:

.. math::

   x^* = \bigl(A^{T}A\bigr)^{-1}A^{T}y.

This is the **Moore–Penrose pseudoinverse** solution, often denoted

.. math::

   x^* = A^{\dagger} y.

#### Geometric Interpretation

The equation :math:`Ax = y` may have no solution if :math:`y \notin \mathrm{Col}(A)`, i.e., if :math:`y` does not lie in the column space of :math:`A`.

Instead, the least-squares solution :math:`x^*` minimizes the distance from :math:`y` to :math:`\mathrm{Col}(A)`:

.. math::

   \lVert A x^* - y \rVert_2 = \min_{x} \lVert Ax - y \rVert_2.

In other words, :math:`A x^*` is the **orthogonal projection** of :math:`y` onto the column space of :math:`A`:

.. math::

   A\,x^* = P_{\mathrm{Col}(A)}\,y,
   \quad\text{where}\quad
   P_{\mathrm{Col}(A)} = A\bigl(A^{T}A\bigr)^{-1}A^{T}.

The matrix :math:`P_{\mathrm{Col}(A)}` is the **orthogonal projection operator** onto :math:`\mathrm{Col}(A)`. It satisfies:

- :math:`P^2 = P` (idempotent),
- :math:`P = P^T` (symmetric),
- :math:`P y = y` iff :math:`y \in \mathrm{Col}(A)`.

This projection-based view emphasizes that **least-squares fitting is not about solving :math:`Ax = y` exactly**, but rather finding the point :math:`Ax^*` in the model space that best approximates :math:`y` in the Euclidean sense.

#### Observability Viewpoint

In this setting, **observability** refers to whether the solution :math:`x^*` is **uniquely determined** by the available data :math:`(A, y)`. This is the case **if and only if** :math:`A` has full column rank—i.e., :math:`\mathrm{rank}(A) = n`.

To make this precise, define the **observability Grammian** for this linear system as:

.. math::

   W := A^{T}A.

This Grammian is positive definite if and only if :math:`A` has full column rank. Then:

- If :math:`W` is invertible: the system is **observable**, and :math:`x^*` is **unique**.
- If :math:`W` is singular: the system is **not observable**, and :math:`x^*` is **not unique**—some directions in state space are hidden to the measurements.

Hence, **least-squares solutions are unique if and only if the Grammian is invertible**, which aligns precisely with the classical definition of observability in control theory.