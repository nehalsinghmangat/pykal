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
^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Motivating Example: Sensor Calibration
======================================

To ground the abstract theory in a concrete engineering task, consider the **calibration** of a simple temperature sensor that outputs a voltage. We assume that, to first approximation, the relationship between true temperature :math:`T` and measured voltage :math:`y` is **linear**:

.. math::

   y = b_0 + b_1\,T + \varepsilon,

where:
- :math:`b_0` is the **offset** (intercept),
- :math:`b_1` is the **sensitivity** (slope),
- :math:`\varepsilon` is a **random noise term**, modeled as a zero-mean Gaussian with variance :math:`\sigma^2`.

Batch Least Squares
-------------------

Suppose we conduct :math:`N` calibration measurements at known temperatures :math:`T_1, T_2, \dots, T_N`, and record the corresponding noisy voltages :math:`y_1, y_2, \dots, y_N`.

We collect these into matrix-vector form:

.. math::

   A \;=\;
   \begin{pmatrix}
     1 & T_1 \\
     1 & T_2 \\
     \vdots & \vdots \\
     1 & T_N
   \end{pmatrix},
   \quad
   x \;=\;
   \begin{pmatrix}
     b_0 \\[3pt] b_1
   \end{pmatrix},
   \quad
   y \;=\;
   \begin{pmatrix}
     y_1 \\[3pt] y_2 \\[3pt] \vdots \\[3pt] y_N
   \end{pmatrix}.

This is precisely the linear model setup discussed earlier:

.. math::

   y = A\,x + \varepsilon,

and our goal is to estimate :math:`x = (b_0, b_1)^T` from the noisy measurements :math:`(A, y)` using **least-squares**, as defined in Eq.~:eq:`eq:batch-def`:

.. math::

   x^* = \arg\min_x\, \lVert A x - y \rVert_2^2.

Once we solve for :math:`x^* = (b_0^*, b_1^*)^T`, we obtain our **calibrated sensor model**:

.. math::

   \hat y(T) = b_0^* + b_1^*\,T.

This matches the solution obtained by solving the **normal equations** (see Eq.~:eq:`eq:batch-solution`), or equivalently, by orthogonally projecting the measurement vector :math:`y` onto the column space of :math:`A` (see Eq.~:eq:`eq:phi` and Eq.~:eq:`eq:gradient`).
In real-time systems (e.g., embedded sensors), calibration must often be updated **sequentially** as new data arrive. This leads naturally to **Sequential Least Squares (RLS)** and eventually to the **Kalman Filter**, both of which generalize the batch least-squares approach above.

We develop these next.

Sequential Leasat Squares
--------------------------------------------------

When calibration measurements arrive one at a time (for example, streaming data from the sensor), we can update our estimate of the offset and sensitivity after each new observation rather than refitting the whole data set.

First, each new measurement at time step :math:`k` consists of
- a temperature :math:`T_k`, and  
- a measured voltage :math:`y_k`.  

We form the **regression row** and **scalar measurement**:

.. math::

   a_k^T = \bigl[\,1,\;T_k\,\bigr],
   \quad
   y_k \in \mathbb{R}.

Initialize your prior estimate and covariance:

.. math::

   x_0 = \begin{pmatrix}b_{0,0}\\[3pt]b_{1,0}\end{pmatrix},
   \quad
   P_0 = \alpha\,I_{2\times2}
   \quad (\alpha\;\text{large}).

Assume measurement noise variance :math:`\sigma^2`. Then at each step **k = 1,2,…**, do:

1. **Compute Kalman gain**  
   .. math::

      K_k = P_{k-1}\,a_k \;\bigl(a_k^T\,P_{k-1}\,a_k + \sigma^2\bigr)^{-1}.

2. **Update estimate**  
   .. math::

      x_k = x_{k-1} + K_k\,\bigl(y_k \;-\; a_k^T\,x_{k-1}\bigr).

3. **Update covariance**  
   .. math::

      P_k = \bigl(I - K_k\,a_k^T\bigr)\,P_{k-1}\,.

After :math:`N` measurements, the final :math:`x_N` equals the batch solution
of Eq.~:eq:`eq:batch-solution`, but with only :math:`O(N\,n^2)` work and constant
memory. This makes sequential RLS ideal for real-time sensor calibration.


For this sensor calibration example, assume the underlying sensor has offset :math:`b_{0}=0.1` and sensitivity :math:`b_{1}=0.05`, with measurement noise variance :math:`\sigma^{2}=0.01`.  Initialize

.. math::

   x_{0} = \begin{pmatrix}0\\0\end{pmatrix},
   \qquad
   P_{0} = 100\,I_{2}.

Then process streaming calibration data \((T_{k},y_{k})\)\:

- **k = 1**  
  :math:`T_{1}=0`, :math:`y_{1}=0.10`.  
  Form

  .. math::

     a_{1}^{T} = [1,\;0],

  then the gain

  .. math::

     K_{1} 
     = \frac{P_{0}\,a_{1}}{a_{1}^{T}\,P_{0}\,a_{1} + \sigma^{2}}
     \approx \begin{pmatrix}0.9999\\[3pt]0\end{pmatrix},

  update

  .. math::

     x_{1} 
     = x_{0} + K_{1}\,(y_{1} - a_{1}^{T}\,x_{0})
     \approx \begin{pmatrix}0.09999\\[3pt]0\end{pmatrix},

  and covariance

  .. math::

     P_{1} 
     = (I - K_{1}\,a_{1}^{T})\,P_{0}
     \approx \begin{pmatrix}0.01 & 0\\[3pt]0 & 100\end{pmatrix}.

- **k = 2**  
  :math:`T_{2}=50`, :math:`y_{2}=2.60`.  
  Form

  .. math::

     a_{2}^{T} = [1,\;50],

  then

  .. math::

     K_{2} 
     = \frac{P_{1}\,a_{2}}{a_{2}^{T}\,P_{1}\,a_{2} + \sigma^{2}}
     \approx \begin{pmatrix}4\times10^{-7}\\[3pt]0.02\end{pmatrix},

  update

  .. math::

     x_{2} 
     = x_{1} + K_{2}\,(y_{2} - a_{2}^{T}\,x_{1})
     \approx \begin{pmatrix}0.09999\\[3pt]0.05000\end{pmatrix},

  which is already very close to the true parameters \(\begin{pmatrix}0.1\\0.05\end{pmatrix}\).

After :math:`N` steps the final :math:`x_{N}` coincides with the batch solution of Eq.~:eq:`eq:batch-solution`, but by updating recursively we only incur :math:`O(N\,n^2)` work and constant memory, ideal for real-time calibration.