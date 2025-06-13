Sensor Calibration: A Motivating Example
========================================

To ground the abstract theory in a concrete engineering task, consider the **calibration** of a simple temperature sensor that outputs a voltage. We assume the relationship between true temperature :math:`T` and measured voltage :math:`y` is approximately **linear**:

.. math::

   y = b_0 + b_1\,T + \varepsilon,

where:

- :math:`b_0` is the **offset** (intercept),
- :math:`b_1` is the **sensitivity** (slope),
- :math:`arepsilon` is a zero-mean Gaussian noise term: :math:`arepsilon \sim \mathcal{N}(0, \sigma^2)`.

Suppose we conduct :math:`N` calibration measurements at known temperatures :math:`T_1, T_2, \dots, T_N` and record the corresponding voltages :math:`y_1, y_2, \dots, y_N`.

This naturally gives rise to a least-squares formulation:

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
     b_0 \\ b_1
   \end{pmatrix},
   \quad
   y \;=\;
   \begin{pmatrix}
     y_1 \\ y_2 \\ \vdots \\ y_N
   \end{pmatrix}.

Then the model is :math:`y = A x + arepsilon` and the batch least-squares solution is:

.. math::

   x^* = (A^T A)^{-1} A^T y.

This yields the estimated parameters :math:`b_0^*, b_1^*`, and the calibrated model:

.. math::

   \hat{y}(T) = b_0^* + b_1^* T.

Observability, Vandermonde Structure, and Conditioning
------------------------------------------------------

The matrix :math:`A` above has a **Vandermonde structure**, and its columns must be linearly independent for the parameters to be uniquely identifiable. If all temperatures :math:`T_k` are the same, the second column of :math:`A` becomes constant and the system becomes unobservable.

Formally, observability corresponds to invertibility of the **observability Grammian**:

.. math::

   W := A^T A.

If :math:`W` is invertible, the system is observable and the parameters are uniquely identifiable. If it is singular (i.e., rank-deficient), then the data cannot resolve all directions in parameter space.

This motivates:

- Choosing diverse :math:`T_k` values (spread across operating range),
- Avoiding clustering or repeated values in calibration,
- Using regularization or RLS for sequential updates.

Sequential Calibration
----------------------

We may wish to update our calibration online, as each new temperature-voltage pair arrives. Let:

.. math::

   a_k^T = \begin{bmatrix}1 & T_k\end{bmatrix},
   \quad
   y_k \in \mathbb{R}

be the regression row and measurement at time :math:`k`. Using **Recursive Least Squares (RLS)**, we recursively update:

1. **Kalman gain:**

.. math::

   K_k = P_{k-1}\,a_k \left(a_k^T P_{k-1} a_k + \sigma^2\right)^{-1}

2. **Estimate:**

.. math::

   x_k = x_{k-1} + K_k \left(y_k - a_k^T x_{k-1}\right)

3. **Covariance:**

.. math::

   P_k = (I - K_k a_k^T) P_{k-1}

Sequential calibration retains the observability viewpoint: the running matrix :math:`M_k = \sum a_i a_i^T` must span :math:`\mathbb{R}^2` for the estimate to converge uniquely.

See also :doc:`sequential_lsq_geometry` for a geometric interpretation of the update.
