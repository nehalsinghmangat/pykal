import numpy as np

# =============================================================================
# 1. Linear systems, NO control input
# =============================================================================

def linear_damped_SHO(t, x):
    """
    Linear, damped harmonic oscillator (no input).

    dx/dt = [ -0.1*x0 + 2*x1,
              -2.0*x0   - 0.1*x1 ]

    Parameters
    ----------
    t : float
        Current time (unused).
    x : array-like, shape (2,)
        State vector [x0, x1].

    Returns
    -------
    list of float
        Derivative [dx0/dt, dx1/dt].

    Examples
    --------
    >>> linear_damped_SHO(0, [1.0, 0.0])
    [-0.1, -2.0]
    """
    x = np.asarray(x)
    return [
        -0.1 * x[0] + 2.0 * x[1],
        -2.0 * x[0] - 0.1 * x[1],
    ]


def linear_3D(t, x):
    """
    Linear 3-state toy system (no input).

    dx/dt = [ -0.1*x0 + 2*x1,
              -2.0*x0 - 0.1*x1,
              -0.3*x2 ]

    Examples
    --------
    >>> linear_3D(0, [1.0, 0.0, 0.0])
    [-0.1, -2.0, -0.0]
    """
    x = np.asarray(x)
    return [
        -0.1 * x[0] + 2.0 * x[1],
        -2.0 * x[0] - 0.1 * x[1],
        -0.3 * x[2],
    ]

# =============================================================================
# 2. Linear systems, WITH control input
# =============================================================================

def linear_damped_SHO_control(t, x, u):
    """
    Linear damped SHO with control input.

    dx/dt = base dynamics + u

    Examples
    --------
    >>> linear_damped_SHO_control(0, [1.0, 0.0], [0.0, 0.0])
    [-0.1, -2.0]
    """
    f0, f1 = linear_damped_SHO(t, x)
    return [f0 + u[0], f1 + u[1]]


def linear_3D_control(t, x, u):
    """
    Linear 3D system with control input.

    Examples
    --------
    >>> linear_3D_control(0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    [-0.1, -2.0, -0.0]
    """
    f = linear_3D(t, x)
    return [f[i] + u[i] for i in range(3)]

# =============================================================================
# 3. Nonlinear systems, NO control input
# =============================================================================

def cubic_damped_SHO(t, x):
    """
    Cubic, damped harmonic oscillator (no input).

    Examples
    --------
    >>> cubic_damped_SHO(0, [1.0, 1.0])
    [1.9, -2.1]
    """
    x = np.asarray(x)
    return [
        -0.1 * x[0]**3 + 2.0 * x[1]**3,
        -2.0 * x[0]**3 - 0.1 * x[1]**3,
    ]


def van_der_pol(t, x, mu=0.5):
    """
    Van der Pol oscillator (no input).

    Examples
    --------
    >>> van_der_pol(0, [1.0, 2.0])
    [2.0, -1.0]
    """
    x0, x1 = x
    return [x1, mu * (1 - x0**2) * x1 - x0]


def duffing(t, x, alpha=0.2, beta=0.05, delta=1):
    """
    Duffing oscillator (no input).

    Examples
    --------
    >>> duffing(0, [1.0, 2.0])
    [2.0, -1.45]
    """
    x0, x1 = x
    return [x1, -alpha * x1 - beta * x0 - delta * x0**3]


def lotka(t, x, a=1, b=10):
    """
    Lotka–Volterra predator-prey model (no input).

    Examples
    --------
    >>> lotka(0, [1.0, 2.0])
    [-19.0, 16.0]
    """
    x0, x1 = x
    return [a * x0 - b * x0 * x1, b * x0 * x1 - 2 * a * x1]


def cubic_oscillator(t, x, p0=-0.1, p1=2, p2=-2, p3=-0.1):
    """
    Generic cubic oscillator (no input).

    Examples
    --------
    >>> cubic_oscillator(0, [1.0, 2.0])
    [15.9, -2.8]
    """
    x0, x1 = x
    return [p0 * x0**3 + p1 * x1**3, p2 * x0**3 + p3 * x1**3]


def rossler(t, x, a=0.2, b=0.2, c=5.7):
    """
    Rössler attractor (no input).

    Examples
    --------
    >>> rossler(0, [1.0, 2.0, 3.0])
    [-5.0, 1.4, -13.9]
    """
    x0, x1, x2 = x
    return [-x1 - x2, x0 + a * x1, b + (x0 - c) * x2]


def hopf(t, x, mu=-0.05, omega=1, A=1):
    """
    Hopf bifurcation model (no input).

    Examples
    --------
    >>> hopf(0, [1.0, 1.0])
    [-3.05, -1.05]
    """
    x0, x1 = x
    r2 = x0**2 + x1**2
    return [mu * x0 - omega * x1 - A * x0 * r2,
            omega * x0 + mu * x1 - A * x1 * r2]


def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    """
    Lorenz system (no input).

    Examples
    --------
    >>> lorenz(0, [1.0, 2.0, 3.0])
    [10.0, 23.0, -6.0]
    """
    x0, x1, x2 = x
    return [sigma * (x1 - x0), x0 * (rho - x2) - x1, x0 * x1 - beta * x2]


def logistic_map(x, mu):
    """
    Logistic map (discrete, no input).

    Examples
    --------
    >>> logistic_map(0.5, 2.0)
    0.5
    """
    return mu * x * (1 - x)

# =============================================================================
# 4. Nonlinear systems, WITH control input
# =============================================================================


def van_der_pol_control(t, x, u, mu=0.5):
    """
    Van der Pol oscillator with control input.

    Examples
    --------
    >>> van_der_pol_control(0, [1.0, 2.0], [0.1, 0.2])
    [2.1, -0.8]
    """
    f0, f1 = van_der_pol(t, x, mu)
    return [f0 + u[0], f1 + u[1]]


def duffing_control(t, x, u, alpha=0.2, beta=0.05, delta=1):
    """
    Duffing oscillator with control input.

    Examples
    --------
    >>> duffing_control(0, [1.0, 2.0], [1.0, 1.0])
    [3.0, -0.45]
    """
    f0, f1 = duffing(t, x, alpha, beta, delta)
    return [f0 + u[0], f1 + u[1]]


def lotka_control(t, x, u, a=1, b=10):
    """
    Lotka–Volterra with control input.

    Examples
    --------
    >>> lotka_control(0, [1.0, 2.0], [1.0, 1.0])
    [-18.0, 17.0]
    """
    f0, f1 = lotka(t, x, a, b)
    return [f0 + u[0], f1 + u[1]]


def cubic_oscillator_control(t, x, u, p0=-0.1, p1=2, p2=-2, p3=-0.1):
    """
    Cubic oscillator with control input.

    Examples
    --------
    >>> cubic_oscillator_control(0, [1.0, 2.0], [0.1, 0.2])
    [16.0, -2.6]
    """
    f0, f1 = cubic_oscillator(t, x, p0, p1, p2, p3)
    return [f0 + u[0], f1 + u[1]]


def rossler_control(t, x, u, a=0.2, b=0.2, c=5.7):
    """
    Rössler attractor with control input.

    Examples
    --------
    >>> rossler_control(0, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    [-4.0, 3.4, -10.9]
    """
    f0, f1, f2 = rossler(t, x, a, b, c)
    return [f0 + u[0], f1 + u[1], f2 + u[2]]


def hopf_control(t, x, u, mu=-0.05, omega=1, A=1):
    """
    Hopf bifurcation model with control input.

    Examples
    --------
    >>> hopf_control(0, [1.0, 1.0], [0.5, 1.0])
    [-2.55, -0.05]
    """
    f0, f1 = hopf(t, x, mu, omega, A)
    return [f0 + u[0], f1 + u[1]]


def lorenz_u(t):
    """
    Sample control input for Lorenz system.

    Examples
    --------
    >>> lorenz_u(0)
    array([0., 0., 0.])
    """
    return np.array([np.sin(2*t)**2, t**2, 0.0])


def lorenz_control(t, x, u_fun, sigma=10, beta=8/3, rho=28):
    """
    Lorenz system with time-varying control input.

    Examples
    --------
    >>> u_fun = lambda t: np.zeros(3)
    >>> lorenz_control(0, [1.0, 2.0, 3.0], u_fun)
    [10.0, 23.0, -6.0]
    """
    f = lorenz(t, x, sigma, beta, rho)
    u = u_fun(t)
    return [f[i] + u[i] for i in range(3)]


def logistic_map_control(x, mu, u):
    """
    Logistic map with scalar control input.

    Examples
    --------
    >>> logistic_map_control(0.5, 2.0, 0.1)
    0.6
    """
    return mu * x * (1 - x) + u


def logistic_map_multicontrol(x, mu, u):
    """
    Logistic map with 2-element control vector.

    Examples
    --------
    >>> logistic_map_multicontrol(0.5, 2.0, [0.1, 0.2])
    0.75
    """
    return mu * x * (1 - x) + u[0] * x + u[1]