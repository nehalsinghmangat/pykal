from typing import Callable, List, Optional
import numpy as np
from numpy.typing import NDArray
from scipy import signal


class Generate:
    @staticmethod
    def constant_signal_function(output_list: List[float]) -> Callable[[], NDArray]:
        def constant_signal() -> NDArray:
            return np.array(output_list).reshape(-1, 1)

        return constant_signal

    @staticmethod
    def heaviside_step_function(
        output_list: List[float], step_time: float = 1.0
    ) -> Callable[[float], NDArray]:
        def heaviside_step(tk: float) -> NDArray:
            if tk < step_time:
                return np.zeros((len(output_list), 1))
            else:
                return np.array(output_list).reshape(-1, 1)

        return heaviside_step

    @staticmethod
    def sinusoidal_function(
        amplitudes: List[float], frequencies: List[float], phases: List[float] = None
    ) -> Callable[[float], NDArray]:
        """
        Generates a multi-dimensional sinusoidal signal.
        """
        if phases is None:
            phases = [0.0] * len(amplitudes)

        def sinusoidal(tk: float) -> NDArray:
            return np.array(
                [
                    amp * np.sin(2 * np.pi * freq * tk + phase)
                    for amp, freq, phase in zip(amplitudes, frequencies, phases)
                ]
            ).reshape(-1, 1)

        return sinusoidal

    @staticmethod
    def sawtooth_function(
        amplitudes: List[float],
        frequencies: List[float],
        widths: List[float] = None,  # Width controls rising vs falling slope
    ) -> Callable[[float], NDArray]:
        if widths is None:
            widths = [0.5] * len(amplitudes)

        def sawtooth(tk: float) -> NDArray:
            return np.array(
                [
                    amp * signal.sawtooth(2 * np.pi * freq * tk, width)
                    for amp, freq, width in zip(amplitudes, frequencies, widths)
                ]
            ).reshape(-1, 1)

        return sawtooth

    @staticmethod
    def square_wave_function(
        amplitudes: List[float],
        frequencies: List[float],
        duty_cycles: List[float] = None,  # Fraction of time signal is high
    ) -> Callable[[float], NDArray]:
        if duty_cycles is None:
            duty_cycles = [0.5] * len(amplitudes)

        def square_wave(tk: float) -> NDArray:
            return np.array(
                [
                    amp * signal.square(2 * np.pi * freq * tk, duty)
                    for amp, freq, duty in zip(amplitudes, frequencies, duty_cycles)
                ]
            ).reshape(-1, 1)

        return square_wave


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


class Signals:
    def __init__(
        self, user_defined_signal: Optional[Callable[[float], NDArray]] = None
    ) -> None:
        self.generate = Generate()
        self.transform = Transform()
        self.user_defined_signal = (
            user_defined_signal  # Optional[Callable[[float], NDArray]]
        )
