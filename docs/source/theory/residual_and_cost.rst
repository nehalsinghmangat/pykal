Minimize the Residual: Least Squares Formulation
================================================

As before, let :math:`f\colon X \to Y` be a smooth function, and suppose we are given :math:`y \in Y`. We define the **residual** as:

.. math::
   :label: eq:residual

   r(x) \;=\; f(x) - y

and the **2-norm** of any vector :math:`v` as:

.. math::
   :label: eq:l2norm

   \lVert v\rVert_2 \;=\; \sqrt{v^\top v}\,.

We wish to find the :math:`x \in X` that minimizes the norm of the residual:

.. math::
   :label: eq:lsq_problem

   \hat{x} \;=\;\arg\min_{x \in X} \lVert f(x) - y\rVert_2^2\,.

That is, even if Eq.~:eq:`eq:funeq` has no exact solution, we look for the :math:`x` whose output :math:`f(x)` is closest to :math:`y` in Euclidean norm. If a true solution does exist, it will also be the least-squares solution.

Cost Function and Gradient
--------------------------

We define the cost function as the squared 2-norm of the residual:

.. math::
   :label: eq:phi

   \phi(x) \;=\; \tfrac12\,\lVert f(x) - y\rVert_2^2.

This choice is convenient because:

- It is differentiable everywhere and non-negative.
- It simplifies the gradient expression using the chain rule:
  
  .. math::
     \nabla\phi(x) \;=\; J(x)^\top(f(x)-y),

  where :math:`J(x)` is the Jacobian matrix of :math:`f`.

- It corresponds to the negative log-likelihood if noise is Gaussian:
  
  .. math::
     y = f(x) + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0,\,\sigma^2 I)

  Then minimizing :math:`\lVert f(x)-y\rVert_2^2` is equivalent to maximizing likelihood.

Gradient-Based Minimization
----------------------------

To minimize :math:`\phi(x)`, we solve:

.. math::
   \nabla \phi(x^*) = 0 \quad\Longrightarrow\quad J(x^*)^\top\,(f(x^*) - y) = 0

This yields a stationary point of the cost. If :math:`f` is linear, the solution is global and unique. If :math:`f` is nonlinear, we apply iterative methods:

- **Gradient Descent:**  
  :math:`x^{(k+1)} = x^{(k)} - \eta\,\nabla\phi(x^{(k)})`
- **Gauss–Newton:**  
  Uses the approximation :math:`J^\top J` for the Hessian.
- **Levenberg–Marquardt:**  
  Interpolates between Gauss–Newton and gradient descent.

These methods are widely used in sensor fitting, system identification, and Kalman filtering variants like the EKF.

Next: We examine the special case where :math:`f(x) = Ax` is linear, leading to a closed-form solution and geometric interpretation.
