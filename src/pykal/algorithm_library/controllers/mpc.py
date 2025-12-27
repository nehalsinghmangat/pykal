import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


class MPC:
    """
    Model Predictive Control (MPC) for constrained optimal control.

    Based on Mayne et al. (2000), "Constrained model predictive control:
    Stability and optimality", Automatica, Vol. 36, No. 6, pp. 789-814.

    MPC solves a finite-horizon optimal control problem at each time step,
    applying only the first control input and resolving (receding horizon).
    """

    @staticmethod
    def f(
        *,
        mpc_state: Tuple[NDArray, NDArray, float],
        xhat_k: NDArray,
        xref_traj: NDArray,
        A: NDArray,
        B: NDArray,
        Q: NDArray,
        R: NDArray,
        P: NDArray,
        N: int,
        u_min: Optional[NDArray] = None,
        u_max: Optional[NDArray] = None,
        x_min: Optional[NDArray] = None,
        x_max: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray, float]:
        """
        Solve the MPC optimization problem for linear discrete-time systems.

        Parameters
        ----------
        mpc_state : Tuple[NDArray, NDArray, float]
            Previous MPC solution ``(u_opt, x_pred, cost)`` containing:
                - ``u_opt`` : optimal control sequence from last solve, shape (N, m)
                - ``x_pred`` : predicted state trajectory, shape (N+1, n)
                - ``cost`` : optimal cost value from last solve
        xhat_k : NDArray
            Current state estimate, shape (n,)
        xref_traj : NDArray
            Reference state trajectory, shape (N+1, n)
        A : NDArray
            State transition matrix, shape (n, n)
        B : NDArray
            Control input matrix, shape (n, m)
        Q : NDArray
            Stage cost state weight (positive semi-definite), shape (n, n)
        R : NDArray
            Stage cost control weight (positive definite), shape (m, m)
        P : NDArray
            Terminal cost weight (positive semi-definite), shape (n, n)
            Typically set to LQR DARE solution for stability
        N : int
            Prediction horizon length
        u_min : NDArray, optional
            Lower bound on control inputs, shape (m,)
        u_max : NDArray, optional
            Upper bound on control inputs, shape (m,)
        x_min : NDArray, optional
            Lower bound on states, shape (n,)
        x_max : NDArray, optional
            Upper bound on states, shape (n,)

        Returns
        -------
        u_opt : NDArray
            Optimal control sequence over horizon, shape (N, m)
        x_pred : NDArray
            Predicted state trajectory, shape (N+1, n)
        cost : float
            Optimal cost value

        Notes
        -----
        The MPC optimization problem at time k is:

        .. math::

            \\min_{u_0, ..., u_{N-1}} \\sum_{i=0}^{N-1} \\left( ||x_i - x_{ref,i}||_Q^2 + ||u_i||_R^2 \\right) + ||x_N - x_{ref,N}||_P^2

        subject to:

        .. math::

            x_{i+1} &= A x_i + B u_i, \\quad i = 0, ..., N-1 \\\\
            x_0 &= \\hat{x}_k \\quad \\text{(initial condition)} \\\\
            u_{min} \\leq u_i &\\leq u_{max}, \\quad i = 0, ..., N-1 \\quad \\text{(input constraints)} \\\\
            x_{min} \\leq x_i &\\leq x_{max}, \\quad i = 1, ..., N \\quad \\text{(state constraints)}

        **Receding Horizon Principle:**
        Only the first control input ``u_0`` is applied. At the next time step,
        the optimization is resolved with updated state and reference.

        **Stability:**
        For unconstrained MPC with terminal cost P equal to the LQR DARE solution,
        the MPC controller is equivalent to LQR and guarantees closed-loop stability.
        With constraints, stability can be ensured via terminal constraint sets.

        **Solver:**
        Uses CVXPY with OSQP backend for fast quadratic programming.
        """
        # Lazy import cvxpy (only when MPC is actually used)
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "cvxpy is required for MPC but not installed. "
                "Install it with: pip install cvxpy>=1.3"
            )

        n = A.shape[0]  # State dimension
        m = B.shape[1]  # Control dimension

        # Flatten current state
        x0 = np.atleast_1d(xhat_k).flatten()

        # Create optimization variables
        x = cp.Variable((N + 1, n))
        u = cp.Variable((N, m))

        # Build cost function
        cost_expr = 0

        for i in range(N):
            # Stage cost
            x_err = x[i, :] - xref_traj[i, :]
            cost_expr += cp.quad_form(x_err, Q) + cp.quad_form(u[i, :], R)

        # Terminal cost
        x_err_terminal = x[N, :] - xref_traj[N, :]
        cost_expr += cp.quad_form(x_err_terminal, P)

        # Build constraints
        constraints = []

        # Initial condition
        constraints.append(x[0, :] == x0)

        # Dynamics constraints
        for i in range(N):
            constraints.append(x[i + 1, :] == A @ x[i, :] + B @ u[i, :])

        # Input constraints
        if u_min is not None:
            for i in range(N):
                constraints.append(u[i, :] >= u_min)

        if u_max is not None:
            for i in range(N):
                constraints.append(u[i, :] <= u_max)

        # State constraints
        if x_min is not None:
            for i in range(1, N + 1):
                constraints.append(x[i, :] >= x_min)

        if x_max is not None:
            for i in range(1, N + 1):
                constraints.append(x[i, :] <= x_max)

        # Formulate and solve QP
        problem = cp.Problem(cp.Minimize(cost_expr), constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Solver failed, return safe fallback (zero control)
                print(f"Warning: MPC solver status: {problem.status}. Using fallback control.")
                u_opt = np.zeros((N, m))
                x_pred = np.tile(x0, (N + 1, 1))
                cost_val = np.inf
            else:
                # Extract solution
                u_opt = u.value
                x_pred = x.value
                cost_val = problem.value

        except Exception as e:
            print(f"Warning: MPC solver exception: {e}. Using fallback control.")
            u_opt = np.zeros((N, m))
            x_pred = np.tile(x0, (N + 1, 1))
            cost_val = np.inf

        return u_opt, x_pred, cost_val

    @staticmethod
    def h(mpc_state: Tuple[NDArray, NDArray, float]) -> NDArray:
        """
        Extract the first control input (receding horizon principle).

        Parameters
        ----------
        mpc_state : Tuple[NDArray, NDArray, float]
            MPC solution ``(u_opt, x_pred, cost)``

        Returns
        -------
        uk : NDArray
            Control input to apply at current time, shape (m,)

        Notes
        -----
        The receding horizon principle:
        - Solve optimization over N-step horizon
        - Apply only first control input ``u_0``
        - At next time step, shift horizon and resolve

        This is the key to MPC's ability to handle constraints and
        time-varying references/disturbances.

        Examples
        --------
        >>> import numpy as np
        >>> u_opt = np.array([[1.0], [0.8], [0.6]])  # 3-step horizon
        >>> x_pred = np.zeros((4, 2))  # Predicted trajectory
        >>> cost = 5.2
        >>> mpc_state = (u_opt, x_pred, cost)
        >>> uk = MPC.h(mpc_state)
        >>> uk
        array([1.])
        """
        u_opt, _, _ = mpc_state

        # Return first control input
        if u_opt.ndim == 1:
            return u_opt
        else:
            return u_opt[0, :].flatten()

    @staticmethod
    def solve_single_step(
        *,
        xk: NDArray,
        xref: NDArray,
        A: NDArray,
        B: NDArray,
        Q: NDArray,
        R: NDArray,
        P: NDArray,
        N: int,
        u_min: Optional[NDArray] = None,
        u_max: Optional[NDArray] = None,
        x_min: Optional[NDArray] = None,
        x_max: Optional[NDArray] = None,
        warm_start: Optional[Tuple[NDArray, NDArray, float]] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Simplified MPC interface for single-step solving.
        
        Parameters
        ----------
        xk : NDArray
            Current state, shape (n,)
        xref : NDArray
            Reference state (constant over horizon), shape (n,)
        A, B, Q, R, P : NDArray
            System matrices and cost weights
        N : int
            Prediction horizon
        u_min, u_max, x_min, x_max : NDArray, optional
            Constraint bounds
        warm_start : Tuple, optional
            Previous MPC solution for warm-starting
            
        Returns
        -------
        u_opt : NDArray
            Optimal control sequence, shape (N, m) or (N,) for scalar
        x_pred : NDArray
            Predicted trajectory, shape (N+1, n)
        """
        # Build constant reference trajectory
        n = len(xk)
        xref_traj = np.tile(xref, (N + 1, 1)) if xref.ndim == 1 else np.tile(xref.reshape(1, -1), (N + 1, 1))
        
        # Initialize or use warm start
        if warm_start is None:
            m = B.shape[1]
            u_init = np.zeros((N, m))
            x_init = np.zeros((N + 1, n))
            cost_init = 0.0
            mpc_state = (u_init, x_init, cost_init)
        else:
            mpc_state = warm_start
        
        # Solve MPC
        u_opt, x_pred, cost = MPC.f(
            mpc_state=mpc_state,
            xhat_k=xk,
            xref_traj=xref_traj,
            A=A,
            B=B,
            Q=Q,
            R=R,
            P=P,
            N=N,
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max
        )

        # Return first control input (receding horizon principle)
        u_first = u_opt[0, 0] if u_opt.ndim == 2 and u_opt.shape[1] == 1 else u_opt[0]
        return u_first, x_pred

    @staticmethod
    def simple_f(
        *,
        xk: NDArray,
        xref: NDArray,
        A: NDArray,
        B: NDArray,
        Q: NDArray,
        R: NDArray,
        P: NDArray,
        N: int,
        u_min: Optional[NDArray] = None,
        u_max: Optional[NDArray] = None,
        x_min: Optional[NDArray] = None,
        x_max: Optional[NDArray] = None,
        mpc_output: Optional[Tuple[NDArray, NDArray]] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Simplified stateless MPC for DynamicalSystem integration.
        
        This is a wrapper around standard_f that handles state initialization.
        Returns (u_opt, x_pred) for compatibility with DynamicalSystem.
        
        Parameters are same as solve_single_step.
        """
        u_opt, x_pred = MPC.solve_single_step(
            xk=xk,
            xref=xref,
            A=A,
            B=B,
            Q=Q,
            R=R,
            P=P,
            N=N,
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max
        )
        
        return u_opt, x_pred
    
    @staticmethod
    def simple_h(mpc_output: Tuple[NDArray, NDArray]) -> NDArray:
        """Extract first control from simple_f output."""
        u_opt, x_pred = mpc_output
        return u_opt


# Module-level aliases for convenience
# Allows usage like: from pykal.algorithm_library.controllers import mpc; mpc.f(...)
f = MPC.f
h = MPC.h
solve_single_step = MPC.solve_single_step
simple_f = MPC.simple_f
simple_h = MPC.simple_h
