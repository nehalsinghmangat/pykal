Observability
=============

In estimation and control theory, observability captures whether the internal state of a system can be inferred from its outputs. While often treated as a structural property of dynamical systems, observability is deeply connected to the invertibility of least-squares problems and the geometry of information.

We explore this from two perspectives:

Statistical Viewpoint: Can the state be uniquely estimated from data?

Control-Theoretic Viewpoint: Can the state be recovered from inputs and outputs?

Statistical Viewpoint: Identifiability in Estimation
----------------------------------------------------

Motivating Example: Can We Calibrate All Parameters?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose we model a sensor’s output as:

.. math::

y_k = b_0 + b_1 T_k + b_2 T_k^2 + \varepsilon_k,

where :math:T_k is a known input (e.g., temperature), :math:y_k is a noisy measurement, and :math:\varepsilon_k \sim \mathcal{N}(0, \sigma^2).

We collect :math:N data points and write:

.. math::

y = A,x + \varepsilon,

where:

.. math::

A = \begin{pmatrix}
1 & T_1 & T_1^2 \
1 & T_2 & T_2^2 \
\vdots & \vdots & \vdots \
1 & T_N & T_N^2
\end{pmatrix},
\quad
x = \begin{pmatrix} b_0 \ b_1 \ b_2 \end{pmatrix},
\quad
y = \begin{pmatrix} y_1 \ y_2 \ \vdots \ y_N \end{pmatrix}.

We wish to solve:

.. math::

x^* = \arg\min_x |A x - y|_2^2.

But is this problem well-posed?

Rank and Identifiability
If :math:A does not have full column rank, the pseudoinverse solution:

.. math::

x^* = A^\dagger y

is not unique. In fact, infinitely many parameter vectors yield the same prediction. This leads to unidentifiability: some directions in parameter space are unobservable from the data.

In general:

The observability of parameters = the rank of the regression matrix :math:A.

The nullspace of :math:A corresponds to unobservable directions:

.. math::

x \in \mathrm{Null}(A) \quad\Rightarrow\quad A x = 0 \quad\Rightarrow\quad \text{no effect on } y.

This means the statistical identifiability of model parameters is equivalent to the injectivity of the map :math:x \mapsto A x.


In summary, from the statistical viewpoint, observability means:

The measurement function :math:f(x) provides enough information to uniquely identify the state (or parameters) from data.

Mathematically, this amounts to the injectivity of the forward model on the set of feasible states.

Control-Theoretic Viewpoint: Dynamical Observability
----------------------------------------------------

Motivating Example: A Hidden Velocity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the linear time-invariant (LTI) system:

.. math::

x_{k+1} &= A x_k + B u_k \
y_k &= C x_k,

with state :math:x_k \in \mathbb{R}^n, control input :math:u_k \in \mathbb{R}^p, and measurements :math:y_k \in \mathbb{R}^m.

Suppose:

.. math::

A = \begin{pmatrix} 1 & \Delta t \ 0 & 1 \end{pmatrix},
\quad
C = \begin{pmatrix} 1 & 0 \end{pmatrix}.

This models position-velocity dynamics with position-only output.

Can we reconstruct both position and velocity?

Observability Matrix
Define the observability matrix:

.. math::

\mathcal{O} = \begin{pmatrix}
C \
C A \
C A^2 \
\vdots \
C A^{n-1}
\end{pmatrix}
\in \mathbb{R}^{n m \times n}.

The system is observable if and only if :math:\mathrm{rank}(\mathcal{O}) = n.

In our example:

.. math::

C = \begin{pmatrix} 1 & 0 \end{pmatrix},
\quad
C A = \begin{pmatrix} 1 & \Delta t \end{pmatrix},

so:

.. math::

\mathcal{O} = \begin{pmatrix}
1 & 0 \
1 & \Delta t
\end{pmatrix}.

This has full rank iff :math:\Delta t \ne 0, implying that both position and velocity can be recovered from measurements over time. Even though velocity is never directly measured, it is observable through the dynamics.

Connection to System Theory
Observability ensures that the internal state can be inferred from inputs and outputs. Formally:

A system is observable if, for any initial state :math:x_0, knowledge of :math:\{y_k, u_k\} for :math:k = 0,\dots,n-1 suffices to determine :math:x_0.

This is a structural property of the matrices :math:(A, C) and is independent of noise or estimation method.

Observability Matrix
--------------------

Linear Systems
^^^^^^^^^^^^^^

For discrete-time linear systems:

.. math::

x_{k+1} = A x_k, \quad y_k = C x_k,

the state is observable iff the observability matrix

.. math::

\mathcal{O} =
\begin{pmatrix}
C \ CA \ CA^2 \ \vdots \ CA^{n-1}
\end{pmatrix}
\in \mathbb{R}^{n \times n}

has full rank.

This tests whether each direction in 
𝑥
x affects the sequence of outputs over time.

Continuous-Time: The Observability Gramian
In the continuous-time linear case:

.. math::

\dot{x}(t) = A x(t), \quad y(t) = C x(t),

the system is observable over 
[
𝑡
0
,
𝑡
1
]
[t 
0
​
,t 
1
​
] iff the observability Gramian

.. math::

W_o(t_0, t_1) = \int_{t_0}^{t_1} e^{A^\top (t - t_0)} C^\top C, e^{A (t - t_0)} dt

is positive definite.

Then, the energy of the output over time reveals the initial condition 
𝑥
(
𝑡
0
)
x(t 
0
​
).

Nonlinear Systems
^^^^^^^^^^^^^^^^^

For a general nonlinear system:

.. math::

\dot{x} = f(x),\quad y = h(x),

define the Lie derivatives:

.. math::

L_f h(x) = \frac{d}{dt} h(x(t)) = \nabla h(x)\cdot f(x),
\quad
L_f^2 h(x) = \nabla (L_f h(x))\cdot f(x), \dots

Build the observability codistribution:

.. math::

\mathcal{O}(x) = \operatorname{span} \bigg{ dh(x),; dL_f h(x),; dL_f^2 h(x),; \dots \bigg}.

Then the system is locally observable near 
𝑥
x iff

.. math::

\operatorname{rank} \mathcal{O}(x) = n.

This is the nonlinear analog of the linear observability matrix: it tests whether the output and its time derivatives carry enough information to reconstruct the full state.

Individual State Observability
------------------------------

In many estimation problems, we are not interested in reconstructing the full state vector, but only in determining the value of one or more **specific** states. This leads to the notion of **individual observability**: whether the value of a **particular component** of the state vector can be uniquely and reliably inferred from available measurements.

Motivating Example: Position Observable, Velocity Ambiguous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a robot moving on a straight track, with state

.. math::

   x = \begin{pmatrix} p \\ v \end{pmatrix}

where :math:`p` is position and :math:`v` is velocity. Suppose its dynamics are:

.. math::

   \dot{x} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} x,
   \quad
   y = \begin{pmatrix} 1 & 0 \end{pmatrix} x.

That is, we **measure position** but not velocity.

We construct the observability matrix:

.. math::

   \mathcal{O} =
   \begin{pmatrix}
     H \\
     H A
   \end{pmatrix}
   =
   \begin{pmatrix}
     1 & 0 \\
     0 & 1
   \end{pmatrix},

which is full rank, so the **full state is observable**.

Now imagine we only observe data over a **short time horizon**. While the observability matrix is technically full rank, the system may not have evolved enough to accurately infer velocity from changes in position. Thus, even if full state observability holds *in theory*, **individual states can have different levels of observability in practice**. We say that **state component** :math:`x_i` is **observable** if, for a given measurement function :math:`y(t)`, input :math:`u(t)`, and system dynamics over :math:`[0,T]`, the value of :math:`x_i(t)` can be uniquely determined.

This does **not** require reconstructing the full state vector.

Row Space Test
^^^^^^^^^^^^^^

Let :math:`\mathcal{O} \in \mathbb{R}^{m \times n}` denote the observability matrix. To determine if a **specific state** :math:`x_i` is observable, test whether its **basis vector** :math:`e_i` lies in the **row space** of :math:`\mathcal{O}`:

.. math::

   x_i \text{ observable } \iff e_i \in \operatorname{RowSpace}(\mathcal{O}).

This is equivalent to:

.. math::

   \lVert P_{\mathcal{O}}\,e_i \rVert \approx 1,

where :math:`P_{\mathcal{O}} = \mathcal{O}^\dagger\,\mathcal{O}` is the orthogonal projector onto the row space of :math:`\mathcal{O}`.

Residual-based score:

.. math::

   \text{ObservabilityScore}(x_i) = \lVert e_i - P_{\mathcal{O}}\,e_i \rVert.

Smaller values imply higher observability.

Nullspace Test (Observability-aware Filtering)
""""""""""""""""""""""""""""""""""""""""""""""

For applications like the Observability-aware Partial-Update Kalman Filter (OPSKF), we use the **nullspace test** to identify unobservable state directions.

Let :math:`o \in \mathbb{R}^{n}` be a candidate update direction (e.g., :math:`\delta x = \hat{x}_k - \hat{x}_{k|k-1}`). Then:

.. math::

   o \text{ unobservable } \iff \lVert \mathcal{O}\,o \rVert < \epsilon,

for some small threshold :math:`\epsilon`. That is, :math:`o` lies in the **nullspace** of the observability matrix.

This test allows us to **suppress** updates in unobservable directions, improving filter stability and estimation accuracy.

Continuum of Observability
""""""""""""""""""""""""""

While classical observability is binary (observable/unobservable), in practice we treat it as a **spectrum**:

- **Strong observability**: residual norm or CRLB is small; state is well estimated  
- **Weak observability**: high uncertainty or sensitivity to noise  
- **Unobservable**: no information from measurements

This continuum arises from:

- The **condition number** of :math:`\mathcal{O}` or :math:`W_O`  
- The **eigenvalue spectrum** of the observability Gramian  
- Projection scores like :math:`\lVert P_{\mathcal{O}}\,e_i \rVert` being **close to 1** (strong) or **close to 0** (weak)

Thus, **observability is not just a property**, but a **measure** of how reliably each state can be inferred. This informs sensor placement, model reduction, and adaptive estimation strategies.

Gramian Diagonal & Fisher Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An alternative approach uses the **observability Gramian**:

.. math::

   W_O = \int_0^T \Phi(t)^T\,H(t)^T\,H(t)\,\Phi(t)\,dt,

where :math:`\Phi(t)` is the state transition matrix.

Then :math:`[W_O]_{ii}` measures the **information** available about state :math:`x_i`. This diagonal entry is also the **inverse of the minimum achievable estimation variance** due to the **Cramér–Rao Lower Bound**.

Cramér–Rao Lower Bound (CRLB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\hat{x}` be an unbiased estimator of the true state :math:`x`, then the **covariance matrix** of any such estimator satisfies:

.. math::

   \operatorname{Cov}(\hat{x}) \;\succeq\; W_O^{-1},

assuming Gaussian measurement noise and linearized dynamics.

This implies:

.. math::

   \operatorname{Var}(\hat{x}_i) \;\ge\; \bigl[W_O^{-1}\bigr]_{ii},

and provides a theoretical limit on how accurately individual states can be estimated.

Thus, a state :math:`x_i` is:

- **Unobservable** if :math:`[W_O]_{ii} = 0`, in which case the variance is unbounded,
- **Weakly observable** if :math:`[W_O]_{ii}` is small,
- **Well observable** if :math:`[W_O]_{ii}` is large.

