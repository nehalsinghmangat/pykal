import numpy as np

# =============================================================================
# Input signal generators (u(t))
# =============================================================================

def u_constant(t, u0):
    """
    Constant control input.

    Examples
    --------
    >>> u_constant(0, [1.0, 2.0])
    array([1., 2.])
    """
    return np.array(u0)


def u_sinusoidal(t, amplitude, frequency, phase=0, offset=0):
    """
    Sinusoidal control input.

    Examples
    --------
    >>> u_sinusoidal(0.25, 1.0, 1.0)
    1.0
    """
    return offset + amplitude * np.sin(2 * np.pi * frequency * t + phase)


def u_step(t, t0, u_before, u_after):
    """
    Step control input: switches at time t0.

    Examples
    --------
    >>> u_step([0, 1, 2], 1, [0], [1])
    array([[0], [1], [1]])
    """
    t = np.asarray(t)
    ub = np.asarray(u_before)
    ua = np.asarray(u_after)
    return np.where(t[..., None] < t0, ub, ua)


def u_bangbang(t, period, u_low, u_high):
    """
    Bang-bang (square-wave) control input.

    Examples
    --------
    >>> u_bangbang([0, 0.5, 1.0], 1.0, 0, 1)
    array([0, 1, 0])
    """
    t = np.asarray(t)
    phase = (t % period) < (period / 2)
    return np.where(phase, u_high, u_low)


def u_ramp(t, slope, intercept=0):
    """
    Ramp (linear) control input.

    Examples
    --------
    >>> u_ramp([0, 1, 2], 1.0, 0)
    array([0, 1, 2])
    """
    return intercept + slope * np.asarray(t)


def u_random(t, low=0.0, high=1.0, size=None, seed=None):
    """
    Random (white noise) control input.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> u_random([0], 0.0, 1.0, size=(1,), seed=0).shape
    (1,)
    """
    rng = np.random.default_rng(seed)
    if size is None:
        size = np.shape(t)
    return rng.uniform(low, high, size)