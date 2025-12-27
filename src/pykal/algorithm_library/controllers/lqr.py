import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from scipy.linalg import solve_discrete_are


class LQR:
    """
    Linear Quadratic Regulator (LQR) for optimal state feedback control.

    Based on Kalman (1960), "Contributions to the theory of optimal control",
    Boletin de la Sociedad Matematica Mexicana, Vol. 5, No. 2, pp. 102-119.

    The LQR controller computes the optimal state feedback gain that minimizes
    a quadratic cost function for linear discrete-time systems.
    """

    @staticmethod
    def compute_gain(
        A: NDArray,
        B: NDArray,
        Q: NDArray,
        R: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the optimal LQR feedback gain by solving the Discrete-Time
        Algebraic Riccati Equation (DARE).

        Parameters
        ----------
        A : NDArray
            State transition matrix, shape (n, n)
        B : NDArray
            Control input matrix, shape (n, m)
        Q : NDArray
            State cost matrix (positive semi-definite), shape (n, n)
            Penalizes deviation from reference state
        R : NDArray
            Control cost matrix (positive definite), shape (m, m)
            Penalizes control effort

        Returns
        -------
        K : NDArray
            Optimal feedback gain matrix, shape (m, n)
        P : NDArray
            Solution to the DARE, shape (n, n)

        Notes
        -----
        The LQR controller minimizes the infinite-horizon quadratic cost:

        .. math::

            J = \\sum_{k=0}^{\\infty} \\left( x_k^T Q x_k + u_k^T R u_k \\right)

        subject to the discrete-time linear dynamics:

        .. math::

            x_{k+1} = A x_k + B u_k

        The optimal control law is:

        .. math::

            u_k = -K (x_k - x_{\\text{ref}})

        where the gain :math:`K` is computed by solving the DARE:

        .. math::

            A^T P A - P - A^T P B (B^T P B + R)^{-1} B^T P A + Q = 0

        and then:

        .. math::

            K = (B^T P B + R)^{-1} B^T P A

        **Ridge Regularization:**
        To ensure numerical stability, a small ridge term :math:`\\epsilon I`
        is added to :math:`R` if it is near-singular.

        Examples
        --------
        >>> import numpy as np
        >>> # Simple double integrator: x[k+1] = [[1, 0.1], [0, 1]] x[k] + [[0], [0.1]] u[k]
        >>> A = np.array([[1.0, 0.1], [0.0, 1.0]])
        >>> B = np.array([[0.0], [0.1]])
        >>> Q = np.eye(2)  # Equal weight on position and velocity
        >>> R = np.array([[1.0]])  # Control effort penalty
        >>> K, P = LQR.compute_gain(A, B, Q, R)
        >>> K.shape
        (1, 2)
        >>> P.shape
        (2, 2)
        >>> # Verify closed-loop stability (all eigenvalues inside unit circle)
        >>> np.all(np.abs(np.linalg.eigvals(A - B @ K)) < 1.0)
        True
        """
        n = A.shape[0]
        m = B.shape[1]

        # Add ridge regularization to R for numerical stability
        ridge = 1e-9
        R_reg = R + ridge * np.eye(m)

        # Solve Discrete-Time Algebraic Riccati Equation (DARE)
        try:
            P = solve_discrete_are(A, B, Q, R_reg)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"Failed to solve DARE. The system may be uncontrollable. "
                f"Original error: {e}"
            )

        # Compute optimal feedback gain
        # K = (B^T P B + R)^{-1} B^T P A
        BT_P_B = B.T @ P @ B
        try:
            K = np.linalg.solve(BT_P_B + R_reg, B.T @ P @ A)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            K = np.linalg.pinv(BT_P_B + R_reg) @ B.T @ P @ A

        return K, P

    @staticmethod
    def f(
        lqr_state: Tuple[NDArray, NDArray],
        xhat_k: NDArray,
        xref_k: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Perform one state update step for the LQR controller.

        For time-invariant LQR, the controller state (optimal gain K and
        DARE solution P) remains constant across time steps.

        Parameters
        ----------
        lqr_state : Tuple[NDArray, NDArray]
            A tuple ``(K, P)`` containing:
                - ``K`` : the optimal feedback gain matrix, shape (m, n)
                - ``P`` : the DARE solution matrix, shape (n, n)
        xhat_k : NDArray
            The current state estimate at time k, shape (n,) or (n, 1)
        xref_k : NDArray
            The reference state at time k, shape (n,) or (n, 1)

        Returns
        -------
        (K, P) : Tuple[NDArray, NDArray]
            The unchanged controller state (time-invariant for infinite-horizon LQR)

        Notes
        -----
        For **time-invariant** infinite-horizon LQR, the optimal gain :math:`K`
        is constant and computed once via ``compute_gain()``. This function
        simply passes through the state unchanged.

        For **time-varying** or finite-horizon LQR (not implemented here), this
        function would update :math:`K_k` and :math:`P_k` at each time step using
        backward Riccati recursion.

        The parameters ``xhat_k`` and ``xref_k`` are included in the signature
        for compatibility with the DynamicalSystem framework but are not used
        in the state update for time-invariant LQR.

        Examples
        --------
        >>> import numpy as np
        >>> A = np.array([[1.0, 0.1], [0.0, 1.0]])
        >>> B = np.array([[0.0], [0.1]])
        >>> Q = np.eye(2)
        >>> R = np.array([[1.0]])
        >>> K, P = LQR.compute_gain(A, B, Q, R)
        >>> lqr_state = (K, P)
        >>> xhat_k = np.array([1.0, 0.5])
        >>> xref_k = np.array([0.0, 0.0])
        >>> K_new, P_new = LQR.standard_f(lqr_state, xhat_k, xref_k)
        >>> np.allclose(K_new, K)  # State unchanged for time-invariant LQR
        True
        >>> np.allclose(P_new, P)
        True
        """
        K, P = lqr_state
        # For time-invariant LQR, state doesn't change
        return (K, P)

    @staticmethod
    def h(
        lqr_state: Tuple[NDArray, NDArray],
        xhat_k: NDArray,
        xref_k: NDArray
    ) -> NDArray:
        """
        Compute the optimal control input from the current controller state.

        Parameters
        ----------
        lqr_state : Tuple[NDArray, NDArray]
            A tuple ``(K, P)`` containing:
                - ``K`` : the optimal feedback gain matrix, shape (m, n)
                - ``P`` : the DARE solution matrix (unused in output computation)
        xhat_k : NDArray
            The current state estimate at time k, shape (n,) or (n, 1)
        xref_k : NDArray
            The reference state at time k, shape (n,) or (n, 1)

        Returns
        -------
        uk : NDArray
            The control input at time k, shape (m,) or (m, 1)

        Notes
        -----
        This function computes the optimal LQR control law:

        .. math::

            u_k = -K (x_k - x_{\\text{ref},k})

        where:
            - :math:`K` is the optimal feedback gain matrix
            - :math:`x_k` is the current state estimate
            - :math:`x_{\\text{ref},k}` is the reference (setpoint) state

        **Negative feedback:** The negative sign ensures that the control input
        opposes the error :math:`(x_k - x_{\\text{ref},k})`, driving the state
        toward the reference.

        **Reference tracking:** When :math:`x_{\\text{ref},k} = 0`, this reduces
        to the standard regulator form :math:`u_k = -K x_k`. For non-zero
        references, the controller tracks :math:`x_{\\text{ref},k}`.

        Examples
        --------
        >>> import numpy as np
        >>> A = np.array([[1.0, 0.1], [0.0, 1.0]])
        >>> B = np.array([[0.0], [0.1]])
        >>> Q = np.eye(2)
        >>> R = np.array([[1.0]])
        >>> K, P = LQR.compute_gain(A, B, Q, R)
        >>> lqr_state = (K, P)
        >>> xhat_k = np.array([1.0, 0.5])  # Current state
        >>> xref_k = np.array([0.0, 0.0])  # Regulate to origin
        >>> uk = LQR.standard_h(lqr_state, xhat_k, xref_k)
        >>> uk.shape
        (1,)
        >>> # Control should oppose the error (negative of K @ error)
        >>> error = xhat_k - xref_k
        >>> np.allclose(uk, -K @ error)
        True
        """
        K, _ = lqr_state

        # Ensure inputs are column vectors for matrix multiplication
        xhat_k = np.atleast_1d(xhat_k).flatten()
        xref_k = np.atleast_1d(xref_k).flatten()

        # Compute tracking error
        error = xhat_k - xref_k

        # Optimal control law: u = -K * (x - x_ref)
        uk = -K @ error

        # Return as 1D array
        return uk.flatten()


# Module-level aliases for convenience
# Allows usage like: from pykal.algorithm_library.controllers import lqr; lqr.f(...)
compute_gain = LQR.compute_gain
f = LQR.f
h = LQR.h
