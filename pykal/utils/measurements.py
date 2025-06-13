import numpy as np

# =============================================================================
# 1. Linear measurement functions, NO control input
# =============================================================================

def h_identity(t, x):
    """
    Identity measurement: observe every state directly.

    y = x

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, shape (n_states,)
        The state vector.

    Returns
    -------
    np.ndarray, shape (n_states,)
        Measured output equal to the state.
    
    Examples
    --------
    >>> h_identity(0.0, [1,2,3])
    array([1, 2, 3])
    """
    return np.asarray(x)


def h_linear_matrix(t, x, H):
    """
    General linear measurement with matrix H.

    y = H @ x

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, shape (n_states,)
        The state vector.
    H : array-like, shape (m, n_states)
        Measurement matrix.

    Returns
    -------
    np.ndarray, shape (m,)
        Linear measurement.

    Examples
    --------
    >>> H = np.array([[1,0,0],[0,1,1]])
    >>> h_linear_matrix(0, [1,2,3], H)
    array([1, 5])
    """
    return np.asarray(H) @ np.asarray(x)


def h_partial(t, x, indices):
    """
    Partial direct measurement of specified state indices.

    y[i] = x[indices[i]]

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, shape (n_states,)
        The state vector.
    indices : sequence of int
        Which state components to observe.

    Returns
    -------
    np.ndarray, shape (len(indices),)
        Subset of the state.

    Examples
    --------
    >>> h_partial(0, [5,6,7], [2,0])
    array([7, 5])
    """
    x = np.asarray(x)
    return x[np.asarray(indices)]


# =============================================================================
# 2. Linear measurement functions, WITH control input
# =============================================================================

def h_identity_control(t, x, u):
    """
    Identity measurement with additive control bias.

    y = x + u

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, shape (n_states,)
        The state vector.
    u : array-like, shape (n_states,)
        Additive bias on each measurement channel.

    Returns
    -------
    np.ndarray, shape (n_states,)
        Biased measurement.

    Examples
    --------
    >>> h_identity_control(0, [1,2], [0.1,-0.2])
    array([ 1. ,  1.8])
    """
    return np.asarray(x) + np.asarray(u)


def h_linear_control(t, x, u, H):
    """
    Linear measurement plus additive control bias.

    y = H @ x + u

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, shape (n_states,)
        The state vector.
    u : array-like, shape (m_measurements,)
        Additive bias on each measurement channel.
    H : array-like, shape (m_measurements, n_states)
        Measurement matrix.

    Returns
    -------
    np.ndarray, shape (m_measurements,)
        Biased linear measurement.

    Examples
    --------
    >>> H = np.eye(2)
    >>> h_linear_control(0, [1,2], [0.5, -0.5], H)
    array([ 0.5,  1.5])
    """
    return np.asarray(H) @ np.asarray(x) + np.asarray(u)


# =============================================================================
# 3. Nonlinear measurement functions, NO control input
# =============================================================================

def h_range(t, x):
    """
    Range measurement from the origin in 2D.

    y = sqrt(x0^2 + x1^2)

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, length >=2
        The state vector; uses x[0], x[1].

    Returns
    -------
    np.ndarray, shape (1,)
        Euclidean distance from origin.

    Examples
    --------
    >>> h_range(0, [3,4])
    array([5.])
    """
    x0, x1 = x[:2]
    return np.array([np.hypot(x0, x1)])


def h_bearing(t, x):
    """
    Bearing (angle) measurement in 2D.

    y = arctan2(x1, x0)

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, length >=2
        The state vector; uses x[0], x[1].

    Returns
    -------
    np.ndarray, shape (1,)
        Angle in radians.

    Examples
    --------
    >>> h_bearing(0, [0,1])
    array([1.57079633])
    """
    x0, x1 = x[:2]
    return np.array([np.arctan2(x1, x0)])


def h_range_bearing(t, x):
    """
    Combined range and bearing in 2D.

    y = [range, bearing]

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, length >=2
        The state vector; uses x[0], x[1].

    Returns
    -------
    np.ndarray, shape (2,)
        [distance, angle].

    Examples
    --------
    >>> h_range_bearing(0, [3,4])
    array([5.       , 0.9272952])
    """
    r = np.hypot(x[0], x[1])
    b = np.arctan2(x[1], x[0])
    return np.array([r, b])


def h_quadratic(t, x):
    """
    Quadratic nonlinear measurement.

    y = [x0^2, x1^2, x0*x1]

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, shape >=2
        The state vector.

    Returns
    -------
    np.ndarray, shape (3,)
        Quadratic terms.

    Examples
    --------
    >>> h_quadratic(0, [2,3])
    array([ 4,  9,  6])
    """
    x = np.asarray(x)
    return np.array([x[0]**2, x[1]**2, x[0]*x[1]])


# =============================================================================
# 4. Nonlinear measurement functions, WITH control input
# =============================================================================

def h_range_control(t, x, u):
    """
    Range measurement with additive scalar bias.

    y = sqrt(x0^2 + x1^2) + u_scalar

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, length >=2
        The state vector.
    u : float
        Additive bias to the range measurement.

    Returns
    -------
    np.ndarray, shape (1,)
        Biased distance.

    Examples
    --------
    >>> h_range_control(0, [3,4], 2.0)
    array([7.])
    """
    r = np.hypot(x[0], x[1])
    return np.array([r + float(u)])


def h_range_bearing_control(t, x, u):
    """
    Range & bearing with separate control biases.

    y0 = range + u[0]
    y1 = bearing + u[1]

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, length >=2
        The state vector.
    u : array-like, length 2
        Biases for [range, bearing].

    Returns
    -------
    np.ndarray, shape (2,)
        Biased [distance, angle].

    Examples
    --------
    >>> h_range_bearing_control(0, [3,4], [1.0, 0.1])
    array([6.       , 1.0272952])
    """
    r = np.hypot(x[0], x[1])
    b = np.arctan2(x[1], x[0])
    return np.array([r + u[0], b + u[1]])


def h_nonlinear_control(t, x, u):
    """
    Example multi-channel nonlinear measurement with input.

    y0 = x0^3 + u[0]
    y1 = sin(x1) + u[1]

    Parameters
    ----------
    t : float
        Current time (ignored).
    x : array-like, length >=2
        The state vector.
    u : array-like, length 2
        Biases for each measurement channel.

    Returns
    -------
    np.ndarray, shape (2,)
        Nonlinear measurement plus bias.

    Examples
    --------
    >>> h_nonlinear_control(0, [2, np.pi/2], [1.0, 0.5])
    array([9.       , 1.5])
    """
    return np.array([x[0]**3 + u[0], np.sin(x[1]) + u[1]])
