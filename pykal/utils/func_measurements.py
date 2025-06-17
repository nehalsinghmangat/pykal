import numpy as np
from numpy.typing import NDArray

# =============================================================================
# 1. Linear measurement functions, NO control input
# =============================================================================

def h_identity(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Identity measurement: observe every state directly.

    >>> h_identity(0.0, np.array([[1.], [2.], [3.]]))
    array([[1.],
           [2.],
           [3.]])
    """
    return x

def h_linear_matrix(t: float, x: NDArray[np.float64], H: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    General linear measurement with matrix H.

    >>> H = np.array([[1.,0.,0.],[0.,1.,1.]])
    >>> h_linear_matrix(0, np.array([[1.], [2.], [3.]]), H)
    array([[1.],
           [5.]])
    """
    return H @ x

def h_partial(t: float, x: NDArray[np.float64], indices: list[int]) -> NDArray[np.float64]:
    """
    Partial direct measurement of specified state indices.

    >>> h_partial(0, np.array([[5.], [6.], [7.]]), [2,0])
    array([[7.],
           [5.]])
    """
    return x[indices, :]

# =============================================================================
# 2. Linear measurement functions, WITH control input
# =============================================================================

def h_identity_control(t: float, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Identity measurement with additive control bias.

    >>> h_identity_control(0, np.array([[1.], [2.]]), np.array([[0.1], [-0.2]]))
    array([[1.1],
           [1.8]])
    """
    return x + u

def h_linear_control(t: float, x: NDArray[np.float64], u: NDArray[np.float64], H: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Linear measurement plus additive control bias.

    >>> H = np.eye(2)
    >>> h_linear_control(0, np.array([[1.], [2.]]), np.array([[0.5], [-0.5]]), H)
    array([[1.5],
           [1.5]])
    """
    return H @ x + u

# =============================================================================
# 3. Nonlinear measurement functions, NO control input
# =============================================================================

def h_range(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Range measurement from the origin in 2D.

    >>> h_range(0, np.array([[3.], [4.]]))
    array([[5.]])
    """
    return np.array([[np.hypot(x[0, 0], x[1, 0])]])

def h_bearing(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Bearing (angle) measurement in 2D.

    >>> h_bearing(0, np.array([[0.], [1.]]))
    array([[1.57079633]])
    """
    return np.array([[np.arctan2(x[1, 0], x[0, 0])]])

def h_range_bearing(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Combined range and bearing in 2D.

    >>> h_range_bearing(0, np.array([[3.], [4.]]))
    array([[5.       ],
           [0.92729522]])
    """
    r = np.hypot(x[0, 0], x[1, 0])
    b = np.arctan2(x[1, 0], x[0, 0])
    return np.array([[r], [b]])

def h_quadratic(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Quadratic nonlinear measurement.

    >>> h_quadratic(0, np.array([[2.], [3.]]))
    array([[4.],
           [9.],
           [6.]])
    """
    return np.array([
        [x[0, 0]**2],
        [x[1, 0]**2],
        [x[0, 0] * x[1, 0]]
    ])

# =============================================================================
# 4. Nonlinear measurement functions, WITH control input
# =============================================================================

def h_range_control(t: float, x: NDArray[np.float64], u: float) -> NDArray[np.float64]:
    """
    Range measurement with additive scalar bias.

    >>> h_range_control(0, np.array([[3.], [4.]]), 2.0)
    array([[7.]])
    """
    r = np.hypot(x[0, 0], x[1, 0])
    return np.array([[r + u]])

def h_range_bearing_control(t: float, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Range & bearing with separate control biases.

    >>> h_range_bearing_control(0, np.array([[3.], [4.]]), np.array([[1.0], [0.1]]))
    array([[6.       ],
           [1.02729522]])
    """
    r = np.hypot(x[0, 0], x[1, 0])
    b = np.arctan2(x[1, 0], x[0, 0])
    return np.array([[r + u[0, 0]], [b + u[1, 0]]])

def h_nonlinear_control(t: float, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Example multi-channel nonlinear measurement with input.

    >>> from numpy import pi
    >>> h_nonlinear_control(0, np.array([[2.], [pi/2]]), np.array([[1.0], [0.5]]))
    array([[9. ],
           [1.5]])
    """
    return np.array([
        [x[0, 0]**3 + u[0, 0]],
        [np.sin(x[1, 0]) + u[1, 0]]
    ])

