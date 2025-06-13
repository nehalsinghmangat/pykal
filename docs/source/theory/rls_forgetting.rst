Recursive Least Squares with Forgetting
=======================================

In many real-time applications, calibration and estimation must be performed **online**, as new data arrive. The **Recursive Least Squares (RLS)** algorithm provides a principled way to update parameter estimates incrementally.

Static Linear Model
-------------------

We assume a static linear model:

.. math::

   y_k = a_k^\top x + \varepsilon_k,

where

- :math:`x \in \mathbb{R}^n`: unknown parameter vector to be estimated
- :math:`a_k \in \mathbb{R}^n`: regressor row at time :math:`k`
- :math:`y_k \in \mathbb{R}`: scalar output
- :math:`\varepsilon_k \sim \mathcal{N}(0, \sigma^2)`: white Gaussian noise

Initialization
--------------

We initialize the estimate and covariance:

.. math::

   x_0 = \text{prior}, \qquad P_0 = \alpha I \quad (\text{e.g., large } \alpha)

RLS Update Equations
--------------------

At each time step :math:`k`, receive :math:`a_k, y_k`, and perform:

1. **Kalman gain**  
   .. math::

      K_k = \frac{P_{k-1} a_k}{\lambda + a_k^\top P_{k-1} a_k}

2. **Estimate update**  
   .. math::

      x_k = x_{k-1} + K_k (y_k - a_k^\top x_{k-1})

3. **Covariance update**  
   .. math::

      P_k = \frac{1}{\lambda} \left(P_{k-1} - K_k a_k^\top P_{k-1} \right)

Here :math:`\lambda \in (0, 1]` is the **forgetting factor**:
- :math:`\lambda = 1`: all data weighted equally
- :math:`\lambda < 1`: recent data weighted more heavily

Interpretation
--------------

RLS performs **incremental orthogonal projections** of the residuals, guided by the precision (inverse covariance). The estimate evolves geometrically along directions that reduce residuals.

The forgetting factor gives **temporal adaptivity**:
- Improves tracking for time-varying parameters
- Fades out outdated information

Connection to Kalman Filter
---------------------------

RLS is a special case of the **Kalman Filter** for a static system:

- System dynamics: :math:`x_k = x_{k-1}`, i.e., constant
- Process noise: zero
- Measurement model: :math:`y_k = a_k^\top x_k + \varepsilon_k`
- Measurement noise: white Gaussian

The RLS update equations match the Kalman filter update when written appropriately.

Observability in RLS
--------------------

The RLS algorithm accumulates information over time in the matrix:

.. math::

   M_k = \sum_{i=1}^k \frac{a_i a_i^\top}{\sigma^2}

This plays the role of the **observability Grammian**. If the regressors :math:`a_k` do not span :math:`\mathbb{R}^n`, then:

- :math:`M_k` is singular → some directions remain unobserved
- The estimate :math:`x_k` is not unique
- The covariance :math:`P_k` does not shrink in unobservable directions

Hence, **observability = persistent excitation**: the data must excite all parameter directions.

Practical Tips
--------------

- Monitor eigenvalues of :math:`P_k` or :math:`M_k` to assess observability
- Use forgetting when tracking time-varying systems
- For correlated or non-Gaussian noise, consider robust or adaptive variants

Next: We explore observability in state-space systems via the **Observability Gramian**.
