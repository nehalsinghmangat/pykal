Vandermonde Systems and Polynomial Sensor Models
================================================

Many sensors exhibit **nonlinear** behavior that is well-approximated by **polynomials** over some operating range. To fit such models, we use **Vandermonde matrices** in a least-squares framework.

Polynomial Model Setup
----------------------

Suppose the sensor output :math:`y` is modeled as a degree-:math:`d` polynomial in input :math:`x`:

.. math::

   y = a_0 + a_1 x + a_2 x^2 + \dots + a_d x^d + \varepsilon

We collect :math:`N` input–output pairs :math:`(x_1, y_1), \dots, (x_N, y_N)`, and construct:

- Vandermonde matrix :math:`A \in \mathbb{R}^{N \times (d+1)}`:

  .. math::

     A =
     \begin{pmatrix}
     1 & x_1 & x_1^2 & \cdots & x_1^d \\
     1 & x_2 & x_2^2 & \cdots & x_2^d \\
     \vdots & \vdots & \vdots & \ddots & \vdots \\
     1 & x_N & x_N^2 & \cdots & x_N^d
     \end{pmatrix}

- Parameter vector:

  .. math::
     x = \begin{pmatrix} a_0 \\ a_1 \\ \vdots \\ a_d \end{pmatrix}

- Output vector:

  .. math::
     y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{pmatrix}

This yields the standard **linear least-squares** problem:

.. math::

   x^* = \arg\min_x \lVert A x - y \rVert_2^2

Conditioning and Observability
------------------------------

The matrix :math:`A^\top A` is the **observability Grammian** in this setting:

.. math::

   W := A^\top A

Observability insight:

- If :math:`W` is invertible: full rank → all polynomial coefficients observable
- If :math:`W` is singular: the data does **not** excite all polynomial terms → unobservable coefficients

Conditioning of :math:`A` is highly sensitive to:

1. **Clustering of inputs**  
   If :math:`x_i` are close together, higher powers become nearly colinear → poor observability

2. **Polynomial degree**  
   Large :math:`d` increases susceptibility to numerical instability

Best Practices
--------------

To ensure reliable fitting and observability:

- Spread inputs :math:`x_i` over the full calibration domain
- Avoid overly high-degree polynomials
- Normalize or scale :math:`x` to improve conditioning (e.g., map :math:`x \in [a,b]` to :math:`[-1,1]`)
- Use **orthogonal polynomials** (e.g., Legendre, Chebyshev) if appropriate

These practices reduce noise amplification and make estimation more robust.

Next: We develop recursive (online) estimation techniques via **Recursive Least Squares** (RLS), which incrementally update estimates as data arrive.

