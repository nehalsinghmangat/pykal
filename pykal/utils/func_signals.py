import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Input signal generators (u(t))
# =============================================================================

def u_constant(t: float, u0: list[float]) -> NDArray[np.float64]:
    """
    Constant control input.

    >>> u_constant(0, [1.0, 2.0])
    array([[1.],
           [2.]])
    """
    return np.array(u0, dtype=np.float64).reshape(-1, 1)


def u_sinusoidal(t: float, amplitude: float, frequency: float, phase: float = 0, offset: float = 0) -> NDArray[np.float64]:
    """
    Sinusoidal control input.

    >>> u_sinusoidal(0.25, 1.0, 1.0)
    array([[1.]])
    """
    val = offset + amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return np.array([[val]], dtype=np.float64)


def u_step(t: list[float] | NDArray[np.float64], t0: float, u_before: list[float], u_after: list[float]) -> NDArray[np.float64]:
    """
    Step control input: switches at time t0.

    >>> u_step([0, 1, 2], 1, [0], [1])
    array([[0.],
           [1.],
           [1.]])
    """
    t = np.asarray(t, dtype=np.float64)
    ub = np.asarray(u_before, dtype=np.float64).reshape(1, -1)
    ua = np.asarray(u_after, dtype=np.float64).reshape(1, -1)
    out = np.where(t[:, None] < t0, ub, ua)
    return out.astype(np.float64)


def u_bangbang(t: list[float] | NDArray[np.float64], period: float, u_low: float, u_high: float) -> NDArray[np.float64]:
    """
    Bang-bang (square-wave) control input.

    >>> u_bangbang([0, 0.5, 1.0], 1.0, 0, 1)
    array([[0.],
           [1.],
           [0.]])
    """
    t = np.asarray(t, dtype=np.float64)
    phase = (t % period) >= (period / 2)
    out = np.where(phase, u_high, u_low)
    return out.reshape(-1, 1).astype(np.float64)


def u_ramp(t: list[float] | NDArray[np.float64], slope: float, intercept: float = 0) -> NDArray[np.float64]:
    """
    Ramp (linear) control input.

    >>> u_ramp([0, 1, 2], 1.0, 0)
    array([[0.],
           [1.],
           [2.]])
    """
    t = np.asarray(t, dtype=np.float64)
    return (intercept + slope * t).reshape(-1, 1)


def u_random(t: list[float] | NDArray[np.float64], low: float = 0.0, high: float = 1.0,
             size: tuple[int, ...] | None = None, seed: int | None = None) -> NDArray[np.float64]:
    """
    Random (white noise) control input.

    >>> u_random([0], 0.0, 1.0, size=(1, 1), seed=0).shape
    (1, 1)
    """
    rng = np.random.default_rng(seed)
    if size is None:
        size = (len(np.atleast_1d(t)), 1)
    return rng.uniform(low, high, size).astype(np.float64)
