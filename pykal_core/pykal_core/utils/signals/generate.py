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
        output: List[float], step_time: float = 1.0
    ) -> Callable[[float], NDArray]:
        def heaviside_step(tk: float) -> NDArray:
            if tk < step_time:
                return np.zeros((len(output), 1))
            else:
                return np.array(output).reshape(-1, 1)

        return heaviside_step

    @staticmethod
    def sinusoidal_function(
        amplitudes: List[float],
        frequencies: List[float],
        phases: Optional[List[float]] = None,
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
