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
