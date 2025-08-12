import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable
from pykal_core.control_system import System


class Jacobian:

    @classmethod
    def _matrix_jacobian_wrt_x_u_t(
        cls,
        sys: System,
        func: Callable,
        xk: NDArray,
        uk: NDArray,
        tk: float,
        epsilon: float = 1e-6,
        include_dt: bool = False,
    ) -> Union[tuple[NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
        """
        Compute empirical Jacobians of a function f(x, u, t) with respect to state, input, and optionally time,
        using central finite differences.

        This is typically used for estimating the Jacobians of a nonlinear **dynamics function** in Kalman filters:
            f : Rⁿ × Rᵐ × R → Rⁿ

        Parameters
        ----------
        func : Callable
            A validated dynamics function of the form f(x, u, t) → Rⁿ, returning a (n, 1) ndarray.
        xk : NDArray
            State vector at time t, of shape (n, 1).
        uk : NDArray
            Input vector at time t, of shape (m, 1).
        tk : float
            Scalar time.
        epsilon : float, optional
            Perturbation size for finite differences. Default is 1e-6.
        include_dt : bool, optional
            Whether to compute the time derivative ∂f/∂t. Default is False.

        Returns
        -------
        Jx : NDArray
            Jacobian ∂f/∂x of shape (n, n).
        Ju : NDArray
            Jacobian ∂f/∂u of shape (n, m).
        Jt : NDArray, optional
            Jacobian ∂f/∂t of shape (n, 1), returned only if `include_dt=True`.

        Examples
        --------
        >>> import numpy as np
        >>> def f(x: NDArray, u: NDArray, t: float) -> NDArray:
        ...     return np.array([
        ...         [x[0, 0] + 2 * u[0, 0] + 0.1 * np.sin(t)],
        ...         [x[1, 0] ** 2 + u[0, 0] + t ** 2]
        ...     ])
        >>> x = np.array([[1.0], [2.0]])
        >>> u = np.array([[0.5]])
        >>> Jx, Ju, Jt = Differentiation.compute_empirical_jacobian(f, x, u, tk=1.0, include_dt=True)
        >>> np.round(Jx, 3)
        array([[1., 0.],
               [0., 4.]])
        >>> np.round(Ju, 3)
        array([[2.],
               [1.]])
        >>> np.round(Jt, 3)
        array([[0.054],
               [2.   ]])
        """
        # Evaluate once to get output dimension (n_output,)
        f_base = sys.safeio.smart_call(func, x=xk, u=uk, t=tk)
        n_output = f_base.shape[0]
        n_states = xk.shape[0]
        n_inputs = uk.shape[0]

        Jx = np.zeros((n_output, n_states))
        Ju = np.zeros((n_output, n_inputs))
        Jt = np.zeros((n_output, 1))  # <-- fix: must match output shape, not state dim!

        # ∂f/∂x
        for i in range(n_states):
            dx = np.zeros_like(xk)
            dx[i, 0] = epsilon
            f_plus = sys.safeio.smart_call(
                func, x=xk + dx, u=uk, t=tk, expected_shape=(n_output, 1)
            )
            f_minus = sys.safeio.smart_call(
                func, x=xk - dx, u=uk, t=tk, expected_shape=(n_output, 1)
            )
            Jx[:, i : i + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂u
        for j in range(n_inputs):
            du = np.zeros_like(uk)
            du[j, 0] = epsilon
            f_plus = sys.safeio.smart_call(
                func, x=xk, u=uk + du, t=tk, expected_shape=(n_output, 1)
            )
            f_minus = sys.safeio.smart_call(
                func, x=xk, u=uk - du, t=tk, expected_shape=(n_output, 1)
            )
            Ju[:, j : j + 1] = (f_plus - f_minus) / (2 * epsilon)

        # ∂f/∂t
        if include_dt:
            f_plus = sys.safeio.smart_call(
                func, x=xk, u=uk, t=tk + epsilon, expected_shape=(n_output, 1)
            )
            f_minus = sys.safeio.smart_call(
                func, x=xk, u=uk, t=tk - epsilon, expected_shape=(n_output, 1)
            )
            Jt = (f_plus - f_minus) / (2 * epsilon)
            return Jx, Ju, Jt

        return Jx, Ju

    @staticmethod
    def wrt_x(
        sys: System,
        func: Callable,
        epsilon: float = 1e-6,
    ) -> Callable:
        """
        Return a closure that computes the Jacobian ∂f/∂x at a given (x, u, t).

        Parameters
        ----------
        func : Callable
            Function f(x, u, t) returning (n×1) ndarray

        Returns
        -------
        Callable
            A function (xk, uk, tk) -> (n×n) ndarray
        """

        def jacobian_x(xk: NDArray, uk: NDArray, tk: float) -> NDArray:
            Jx, _ = Jacobian._matrix_jacobian_wrt_x_u_t(
                sys, func, xk, uk, tk, epsilon=epsilon
            )
            return Jx

        return jacobian_x

    @staticmethod
    def wrt_u(
        sys: System,
        func: Callable,
        epsilon: float = 1e-6,
    ) -> Callable:
        """
        Return a closure that computes the Jacobian ∂f/∂u at a given (x, u, t).

        Parameters
        ----------
        func : Callable
            Function f(x, u, t) returning (n×1) ndarray

        Returns
        -------
        Callable
            A function (xk, uk, tk) -> (n×m) ndarray
        """

        def jacobian_u(xk: NDArray, uk: NDArray, tk: float) -> NDArray:
            _, Ju = Jacobian._matrix_jacobian_wrt_x_u_t(
                sys, func, xk, uk, tk, epsilon=epsilon
            )
            return Ju

        return jacobian_u

    @staticmethod
    def wrt_t(
        sys: System,
        func: Callable,
        epsilon: float = 1e-6,
    ) -> Callable:
        """
        Return a closure that computes the Jacobian ∂f/∂t at a given (x, u, t).

        Parameters
        ----------
        func : Callable
            Function f(x, u, t) returning (n×1) ndarray

        Returns
        -------
        Callable
            A function (xk, uk, tk) -> (n×1) ndarray
        """

        def jacobian_t(xk: NDArray, uk: NDArray, tk: float) -> NDArray:
            _, _, Jt = Jacobian._matrix_jacobian_wrt_x_u_t(
                sys, func, xk, uk, tk, epsilon=epsilon, include_dt=True
            )
            return Jt

        return jacobian_t
