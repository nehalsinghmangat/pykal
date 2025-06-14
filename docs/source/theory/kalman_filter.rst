Kalman Filter
=============

The Kalman Filter generalizes sequential least-squares to the case where the state evolves over time. Instead of estimating a static parameter vector, we now seek to estimate a dynamical state :math:x_k governed by process and measurement models:

.. math::
x_k &= f(x_{k-1}, u_{k-1}) + w_{k-1} \
y_k &= h(x_k) + v_k

where:

:math:x_k is the hidden state at time :math:k,

:math:u_k is a known control input,

:math:y_k is the observed measurement,

:math:w_k \sim \mathcal{N}(0, Q) is process noise,

:math:v_k \sim \mathcal{N}(0, R) is measurement noise.

Motivating Example: Tracking a Moving Object
--------------------------------------------

Suppose a drone is flying in a straight line. We want to estimate its position and velocity using noisy position measurements at discrete time steps.

We model its dynamics and observations as:

.. math::
x_k &= A x_{k-1} + w_{k-1}, \
y_k &= H x_k + v_k,

with:

.. math::
A = \begin{pmatrix}1 & \Delta t\ 0 & 1\end{pmatrix},
\quad
H = \begin{pmatrix}1 & 0\end{pmatrix},

where the state is :math:x_k = \begin{pmatrix}\text{position}\\ \text{velocity}\end{pmatrix}. This is a classic linear Gaussian state-space model.

Connection to Least-Squares and RLS
-----------------------------------

Like batch least-squares, the Kalman filter computes minimum-variance estimates using linear algebra.

Like sequential least-squares, it recursively updates the estimate as new measurements arrive.

But unlike RLS, the Kalman filter:

Handles time-varying dynamics and controls,

Incorporates process noise to model state uncertainty,

Maintains an evolving covariance matrix for belief updates.

Kalman Filter Algorithm (Linear Case)
-------------------------------------

Let the system evolve as:

.. math::
x_k &= A x_{k-1} + B u_{k-1} + w_{k-1}, \
y_k &= H x_k + v_k,

with known matrices :math:A, B, H, and noise covariances :math:Q, R. Then the Kalman Filter proceeds as:

Initialization:

.. math::
x_0 \sim \mathcal{N}(\hat{x}_0, P_0)

At each time step:

Predict:

.. math::
\hat{x}{k|k-1} &= A,\hat{x}{k-1} + B,u_{k-1} \
P_{k|k-1} &= A,P_{k-1},A^T + Q

Update:

.. math::
K_k &= P_{k|k-1},H^T,(H,P_{k|k-1},H^T + R)^{-1} \quad\text{(Kalman gain)} \
\hat{x}k &= \hat{x}{k|k-1} + K_k,(y_k - H,\hat{x}{k|k-1}) \
P_k &= (I - K_k,H),P{k|k-1}

This is a recursive optimal estimator when the system is linear and Gaussian. Each update is a projection of the measurement onto the estimated state—just as in the orthogonal projection interpretation of least-squares!

Handling Nonlinear Systems: The EKF
-----------------------------------

In many applications (e.g., robotics, navigation), the dynamics and measurements are nonlinear:

.. math::
x_k &= f(x_{k-1}, u_{k-1}) + w_{k-1}, \
y_k &= h(x_k) + v_k.

The Extended Kalman Filter (EKF) linearizes these equations at each time step using the Jacobians:

.. math::
F_k &= \frac{\partial f}{\partial x}(x_{k-1}, u_{k-1}), \
H_k &= \frac{\partial h}{\partial x}(x_{k|k-1}).

Then the EKF follows the same predict-update steps, replacing :math:A, H with :math:F_k, H_k. This mirrors the nonlinear least-squares gradient approach discussed earlier.

Partial-Update Kalman Filter (PUKF)
-----------------------------------

In many practical systems, only part of the state is observable or relevant at each step. Updating the full state vector may:

Introduce unnecessary computation,

Overfit to noise in unobservable directions,

Corrupt well-estimated state components with uncertain observations.

The Partial-Update Kalman Filter (PUKF) addresses this by selectively updating only those components of the state that are observable (or sensitive) to the current measurement.

Let :math:\beta_k \in \mathbb{R}^{n \times n} be a (possibly time-varying) update mask matrix that determines which state components are affected by the Kalman gain. Then the update step becomes:

.. math::
x_k &= x_{k|k-1} + \beta_k,K_k,(y_k - H x_{k|k-1}), \
P_k &= (I - \beta_k K_k H) P_{k|k-1} (I - \beta_k K_k H)^T + \beta_k K_k R K_k^T \beta_k^T.

Special cases:
^^^^^^^^^^^^^^

:math:\beta_k = I: full update (reduces to regular Kalman filter),

:math:\beta_k = 0: no update (prediction only),

Diagonal :math:\beta_k: update selected states only.

The choice of :math:\beta_k allows targeted, robust updates that reflect confidence in the information content of the measurement.

Connection to Projection
^^^^^^^^^^^^^^^^^^^^^^^^
The use of :math:\beta_k modifies the direction in which the measurement residual is projected. This corresponds to choosing a subspace of the state space to correct, similar to a restricted orthogonal projection in linear algebra.

If we think of :math:x_k as a point in :math:\mathbb{R}^n, then the PUKF projects the residual only onto the subspace spanned by the “active” directions in :math:\beta_k.

Observability-Aware Partial Update Kalman Filter (OPSKF)
--------------------------------------------------------

The PUKF is powerful, but it still requires user-defined update masks :math:\beta_k. In practice, we would like to select this matrix automatically based on the current observability of the system.

The Observability-aware PUKF (OPSKF) uses the local observability Gramian or its proxies to determine which state directions are actually informed by the data.

Null-Space Method (OPSKF-Null)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:\mathcal{O}_k denote the local observability matrix:

.. math::
\mathcal{O}_k = \begin{pmatrix}
H_k \
H_k F_k \
\vdots \
H_k F_k^{n-1}
\end{pmatrix}.

Compute its null space, and project the residual only onto directions orthogonal to it:

.. math::
\beta_k = I - \Pi_{\mathrm{Null}(\mathcal{O}_k)}.

Here, :math:\Pi_{\mathrm{Null}(\mathcal{O}_k)} is the projection onto the null space of the observability matrix. This ensures that state directions that are unobservable (up to the current time) are not corrected.

Stochastic Approximation (OPSKF-Stochastic)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, let :math:P_k be the current state covariance. Use its eigenvalue decomposition:

.. math::
P_k = V \Lambda V^T.

Then define :math:\beta_k to project onto the subspace associated with low-uncertainty, high-observability directions. For example, threshold the eigenvalues:

.. math::
\beta_k = \sum_{i:,\lambda_i \le \tau},v_i v_i^T,

where :math:v_i is the i-th eigenvector and :math:\lambda_i is its corresponding eigenvalue.

This method trades exact null-space computation for a probabilistic proxy for observability, which is often more robust in nonlinear, noisy systems.

