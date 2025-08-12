from typing import Callable, List, Optional
import numpy as np
from numpy.typing import NDArray
from scipy import signal

class Transform:
    @staticmethod
    def signals(
        func: Callable[..., NDArray], *signals: Callable[[float], NDArray]
    ) -> Callable[[float], NDArray]:
        """
        Compose a transformation function over time-dependent signal functions.

        Parameters
        ----------
        func : Callable[..., NDArray]
            A function that takes one or more NDArray inputs and returns an NDArray.
        *signals : Callable[[float], NDArray]
            Time-dependent signal functions.

        Returns
        -------
        Callable[[float], NDArray]
            A function of time that applies `func` to the outputs of the input signal functions.
        """

        def composed_signal(t: float) -> NDArray:
            return func(*[s(t) for s in signals])

        return composed_signal

    # --- Binary Operations ---
    @staticmethod
    def add(a: NDArray, b: NDArray) -> NDArray:
        return a + b

    @staticmethod
    def subtract(a: NDArray, b: NDArray) -> NDArray:
        return a - b

    @staticmethod
    def multiply(a: NDArray, b: NDArray) -> NDArray:
        return a * b

    @staticmethod
    def divide(a: NDArray, b: NDArray) -> NDArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.true_divide(a, b)
            result[~np.isfinite(result)] = 0.0
        return result

    @staticmethod
    def dot(a: NDArray, b: NDArray) -> NDArray:
        return np.dot(a, b)

    # --- Unitary Operations ---
    @staticmethod
    def negate(a: NDArray) -> NDArray:
        return -a

    @staticmethod
    def abs(a: NDArray) -> NDArray:
        return np.abs(a)

    @staticmethod
    def square(a: NDArray) -> NDArray:
        return np.square(a)

    @staticmethod
    def sqrt(a: NDArray) -> NDArray:
        return np.sqrt(np.clip(a, 0, None))

    @staticmethod
    def norm(a: NDArray) -> NDArray:
        return np.array([[np.linalg.norm(a)]])

    @staticmethod
    def clip(a: NDArray, min_val: float, max_val: float) -> NDArray:
        return np.clip(a, min_val, max_val)

    # --- Nonlinear Activations ---
    @staticmethod
    def relu(a: NDArray) -> NDArray:
        return np.maximum(0, a)

    @staticmethod
    def tanh(a: NDArray) -> NDArray:
        return np.tanh(a)

    @staticmethod
    def sigmoid(a: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-a))

    # --- Thresholding / Logical ---
    @staticmethod
    def threshold(a: NDArray, thresh: float) -> NDArray:
        return (a >= thresh).astype(float)

    @staticmethod
    def greater_than(a: NDArray, b: NDArray) -> NDArray:
        return (a > b).astype(float)

    @staticmethod
    def less_than(a: NDArray, b: NDArray) -> NDArray:
        return (a < b).astype(float)
