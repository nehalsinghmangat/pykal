Motivating Example: Sensor Calibration
======================================

To ground the abstract theory in a concrete engineering task, consider the **calibration** of a simple temperature sensor that outputs a voltage. We assume that, to first approximation, the relationship between true temperature :math:`T` and measured voltage :math:`y` is **linear**:

.. math::

   y = b_0 + b_1\,T + \varepsilon,

where:
- :math:`b_0` is the **offset** (intercept),
- :math:`b_1` is the **sensitivity** (slope),
- :math:`\varepsilon` is a **random noise term**, modeled as a zero-mean Gaussian with variance :math:`\sigma^2`.

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

Noise and Optimality
^^^^^^^^^^^^^^^^^^^^

The noise term :math:`\varepsilon` reflects sensor and system imperfections. If we assume

.. math::

   \varepsilon \sim \mathcal{N}(0,\,\sigma^2 I),

then the least-squares solution :math:`x^*` is also the **maximum likelihood estimate** (MLE). This is one of the fundamental motivations for using least-squares in calibration and filtering problems.

Observability and the Grammian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As discussed in the orthogonal projection section, we say the calibration problem is **observable** if the solution :math:`x^*` is **unique**. For linear least-squares, this is the case when the **observability Grammian**

.. math::

   W = A^T A

is **invertible**. Geometrically, this means that the columns of :math:`A` span a full 2D subspace in :math:`\mathbb{R}^N`, and algebraically, that the temperature values :math:`T_1,\dots,T_N` are **not all equal**.

If :math:`W` is singular (i.e., rank-deficient), then multiple values of :math:`(b_0, b_1)` are consistent with the same observed data — so we cannot uniquely recover the parameters.

In this context, **observability = parameter identifiability**: we can only estimate a direction in parameter space if the data contain enough variation in temperature.

Vandermonde Matrix Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The matrix :math:`A` here has a **Vandermonde** structure:

.. math::

   A =
   \begin{pmatrix}
   1 & T_1 \\
   1 & T_2 \\
   \vdots & \vdots \\
   1 & T_N
   \end{pmatrix}.

This structure generalizes to polynomial regression, where higher powers of :math:`T` are added as columns. The Vandermonde matrix is well-conditioned when the temperature points :math:`T_k` are well-spread, but becomes ill-conditioned when they cluster.

To ensure numerical and statistical robustness, we should:
- Use well-separated calibration temperatures,
- Avoid fitting high-degree polynomials,
- Scale or normalize the inputs where appropriate.

Connection to Filtering
^^^^^^^^^^^^^^^^^^^^^^^

In real-time systems (e.g., embedded sensors), calibration must often be updated **sequentially** as new data arrive. This leads naturally to **Recursive Least Squares (RLS)** and eventually to the **Kalman Filter**, both of which generalize the batch least-squares approach above.

We develop these next.

Vandermonde Systems and Polynomial Sensor Models
------------------------------------------------
Many real-world sensors exhibit nonlinear behavior. When the underlying input-output relationship is smooth and approximately polynomial, we can use Vandermonde matrices for calibration.

Suppose a sensor output :math:y is a polynomial function of an input :math:x:

.. math::
y = a_0 + a_1 x + a_2 x^2 + \dots + a_d x^d + \varepsilon

We collect :math:N input-output pairs :math:(x_i, y_i), and construct a Vandermonde matrix:

.. math::
A =
\begin{pmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^d \
1 & x_2 & x_2^2 & \cdots & x_2^d \
\vdots & \vdots & \vdots & \ddots & \vdots \
1 & x_N & x_N^2 & \cdots & x_N^d \
\end{pmatrix},
\quad
x = \begin{pmatrix} a_0 \ a_1 \ \vdots \ a_d \end{pmatrix},
\quad
y = \begin{pmatrix} y_1 \ y_2 \ \vdots \ y_N \end{pmatrix}

This yields a standard least-squares problem:

.. math::
x^* = \arg\min_x \lVert A x - y \rVert_2^2

Observability Insight: The matrix :math:A^\top A becomes poorly conditioned if the input data :math:x_i are clustered or if :math:d is large. This is analogous to poor observability: we cannot resolve all coefficients uniquely if the data doesn’t excite the full polynomial space.

Sequential Calibration via Recursive Least Squares
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

Sequential (Recursive) Least Squares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When rows :math:`a_k^T` of :math:`A` and scalars :math:`y_k` arrive one at a time, one can update the estimate without refitting all data.

We maintain at step :math:`k`:
- estimate :math:`x_{k}`,
- covariance :math:`P_{k}`.

Initialize  
:math:`x_{0}` (prior) and :math:`P_{0}` (large).

At step :math:`k`, observe  
:math:`a_k^T \in \mathbb{R}^{1\times n}`, :math:`y_k\in\mathbb{R}`.  

1. **Predict** (here trivial, static model):  
   :math:`x_{k\mid k-1} = x_{k-1}`,  
   :math:`P_{k\mid k-1} = P_{k-1}`.

2. **Compute gain:**  
   .. math::
      :label: eq:rls-gain

      K_k 
      = P_{k\mid k-1}\,a_k\,
        \bigl(a_k^T\,P_{k\mid k-1}\,a_k + \sigma^2\bigr)^{-1},

   where :math:`\sigma^2` is the assumed measurement variance.

3. **Update estimate and covariance:**  
   .. math::
      x_{k} 
      = x_{k\mid k-1} + K_k\,(y_k - a_k^T\,x_{k\mid k-1}),
      \qquad
      P_{k} 
      = (I - K_k\,a_k^T)\,P_{k\mid k-1}.\label{eq:rls-update}

This ensures that after :math:`m` steps one recovers the batch solution in Eq.~\eqref{eq:batch-solution}.

**Example.**  
Use the same data as above, observing rows in order:

1. :math:`a_1^T=(1,1), y_1=2`.  
   With large :math:`P_0`, the first update essentially fits the line through that point.

2. :math:`a_2^T=(1,2), y_2=3` → refine.

3. :math:`a_3^T=(1,3), y_3=5` → final :math:`x_3=(1,1)`.

At each step the update formulas \[eq:rls-gain\]–\[eq:rls-update\] project the new residual onto
the current parameter–covariance ellipse, analogous to an orthogonal projection in the batch case.

This **recursive** approach is ideal for streaming data or when :math:`m` is very large.

Example: Sequential Sensor Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assume the underlying sensor has offset :math:`b_{0}=0.1` and sensitivity :math:`b_{1}=0.05`, with measurement noise variance :math:`\sigma^{2}=0.01`.  Initialize

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
