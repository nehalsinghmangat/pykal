import numpy as np
from numpy.typing import NDArray

def linear_damped_SHO(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Linear, damped harmonic oscillator (no input).

    dx/dt = [ -0.1*x0 + 2*x1,
              -2.0*x0   - 0.1*x1 ]

    Examples
    --------
    >>> linear_damped_SHO(0, np.array([[1.0], [0.0]]))
    array([[-0.1],
           [-2. ]])
    """
    return np.array([
        [-0.1 * x[0, 0] + 2.0 * x[1, 0]],
        [-2.0 * x[0, 0] - 0.1 * x[1, 0]]
    ])

def linear_3D(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Linear 3-state toy system (no input).

    Examples
    --------
    >>> linear_3D(0, np.array([[1.0], [0.0], [0.0]]))
    array([[-0.1],
           [-2. ],
           [-0. ]])
    """
    return np.array([
        [-0.1 * x[0, 0] + 2.0 * x[1, 0]],
        [-2.0 * x[0, 0] - 0.1 * x[1, 0]],
        [-0.3 * x[2, 0]]
    ])

def cubic_damped_SHO(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Cubic, damped harmonic oscillator (no input).

    Examples
    --------
    >>> cubic_damped_SHO(0, np.array([[1.0], [1.0]]))
    array([[ 1.9],
           [-2.1]])
    """
    return np.array([
        [-0.1 * x[0, 0]**3 + 2.0 * x[1, 0]**3],
        [-2.0 * x[0, 0]**3 - 0.1 * x[1, 0]**3]
    ])

def van_der_pol(t: float, x: NDArray[np.float64], mu: float = 0.5) -> NDArray[np.float64]:
    """
    Van der Pol oscillator (no input).

    Examples
    --------
    >>> van_der_pol(0, np.array([[1.0], [2.0]]))
    array([[ 2.],
           [-1.]])
    """
    x0, x1 = x[0, 0], x[1, 0]
    return np.array([
        [x1],
        [mu * (1 - x0**2) * x1 - x0]
    ])

def duffing(t: float, x: NDArray[np.float64], alpha=0.2, beta=0.05, delta=1) -> NDArray[np.float64]:
    """
    Duffing oscillator (no input).

    Examples
    --------
    >>> duffing(0, np.array([[1.0], [2.0]]))
    array([[ 2.  ],
           [-1.45]])
    """
    x0, x1 = x[0, 0], x[1, 0]
    return np.array([
        [x1],
        [-alpha * x1 - beta * x0 - delta * x0**3]
    ])

def lotka(t: float, x: NDArray[np.float64], a=1, b=10) -> NDArray[np.float64]:
    """
    Lotka–Volterra predator-prey model (no input).

    Examples
    --------
    >>> lotka(0, np.array([[1.0], [2.0]]))
    array([[-19.],
           [ 16.]])
    """
    x0, x1 = x[0, 0], x[1, 0]
    return np.array([
        [a * x0 - b * x0 * x1],
        [b * x0 * x1 - 2 * a * x1]
    ])

def cubic_oscillator(t: float, x: NDArray[np.float64], p0=-0.1, p1=2, p2=-2, p3=-0.1) -> NDArray[np.float64]:
    """
    Generic cubic oscillator (no input).

    Examples
    --------
    >>> cubic_oscillator(0, np.array([[1.0], [2.0]]))
    array([[15.9],
           [-2.8]])
    """
    x0, x1 = x[0, 0], x[1, 0]
    return np.array([
        [p0 * x0**3 + p1 * x1**3],
        [p2 * x0**3 + p3 * x1**3]
    ])

def rossler(t: float, x: NDArray[np.float64], a=0.2, b=0.2, c=5.7) -> NDArray[np.float64]:
    """
    Rössler attractor (no input).

    Examples
    --------
    >>> rossler(0, np.array([[1.0], [2.0], [3.0]]))
    array([[ -5. ],
           [  1.4],
           [-13.9]])
    """
    x0, x1, x2 = x[0, 0], x[1, 0], x[2, 0]
    return np.array([
        [-x1 - x2],
        [x0 + a * x1],
        [b + (x0 - c) * x2]
    ])

def hopf(t: float, x: NDArray[np.float64], mu=-0.05, omega=1, A=1) -> NDArray[np.float64]:
    """
    Hopf bifurcation model (no input).

    Examples
    --------
    >>> hopf(0, np.array([[1.0], [1.0]]))
    array([[-3.05],
           [-1.05]])
    """
    x0, x1 = x[0, 0], x[1, 0]
    r2 = x0**2 + x1**2
    return np.array([
        [mu * x0 - omega * x1 - A * x0 * r2],
        [omega * x0 + mu * x1 - A * x1 * r2]
    ])

def lorenz(t: float, x: NDArray[np.float64], sigma=10, beta=8/3, rho=28) -> NDArray[np.float64]:
    """
    Lorenz system (no input).

    Examples
    --------
    >>> lorenz(0, np.array([[1.0], [2.0], [3.0]]))
    array([[10.],
           [23.],
           [-6.]])
    """
    x0, x1, x2 = x[0, 0], x[1, 0], x[2, 0]
    return np.array([
        [sigma * (x1 - x0)],
        [x0 * (rho - x2) - x1],
        [x0 * x1 - beta * x2]
    ])
