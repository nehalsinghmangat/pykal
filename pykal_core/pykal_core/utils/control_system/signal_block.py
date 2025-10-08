import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional
from scipy import signal


class SignalFunctions:

    @staticmethod
    def _resolve_frequency(
        frequency: Optional[float], period: Optional[float]
    ) -> float:
        if frequency is not None and period is not None:
            raise ValueError("Specify either frequency or period, not both.")
        if frequency is None and period is None:
            raise ValueError("Must provide either frequency or period.")
        return frequency if frequency is not None else 1.0 / period

    @staticmethod
    def constant_signal_function(value: float) -> Callable[[], np.ndarray]:
        """
        Constant (time-independent) signal returning a 1D array of length 1.
        """

        def constant_signal() -> np.ndarray:
            return np.array([value], dtype=float)

        return constant_signal

    @staticmethod
    def make_heaviside_step_function(
        value: float, step_time: float = 1.0
    ) -> Callable[[float], np.ndarray]:
        """
        Heaviside step: 0 before step_time, constant value after.
        """

        def heaviside_step(tk: float) -> np.ndarray:
            return np.array([0.0 if tk < step_time else value], dtype=float)

        return heaviside_step

    @staticmethod
    def make_sinusoidal_function(
        amplitude: float,
        frequency: Optional[float] = None,
        period: Optional[float] = None,
        phase: float = 0.0,
    ) -> Callable[[float], NDArray]:
        """
        Sinusoidal signal:
        s(t) = amplitude * sin(2π f t + phase)

        Parameters
        ----------
        amplitude : float
            Signal amplitude.
        frequency : float, optional
            Frequency in Hz.
        period : float, optional
            Period in seconds.
        phase : float, optional
            Phase offset in radians.
        """
        f = SignalFunctions._resolve_frequency(frequency, period)

        def sinusoidal(tk: float) -> NDArray:
            return np.array(
                [amplitude * np.sin(2 * np.pi * f * tk + phase)], dtype=float
            )

        return sinusoidal

    @staticmethod
    def make_sawtooth_function(
        amplitude: float,
        frequency: Optional[float] = None,
        period: Optional[float] = None,
        width: float = 0.5,
    ) -> Callable[[float], NDArray]:
        """
        Sawtooth wave:
        s(t) = amplitude * sawtooth(2π f t, width)

        Parameters
        ----------
        amplitude : float
            Signal amplitude.
        frequency : float, optional
            Frequency in Hz.
        period : float, optional
            Period in seconds.
        width : float, optional
            Controls rising vs falling slope (0–1).
        """
        f = SignalFunctions._resolve_frequency(frequency, period)

        def sawtooth(tk: float) -> NDArray:
            return np.array(
                [amplitude * signal.sawtooth(2 * np.pi * f * tk, width)], dtype=float
            )

        return sawtooth

    @staticmethod
    def make_square_wave_function(
        amplitude: float,
        frequency: Optional[float] = None,
        period: Optional[float] = None,
        duty_cycle: float = 0.5,
    ) -> Callable[[float], NDArray]:
        """
        Square wave:
        s(t) = amplitude * square(2π f t, duty)

        Parameters
        ----------
        amplitude : float
            Signal amplitude.
        frequency : float, optional
            Frequency in Hz.
        period : float, optional
            Period in seconds.
        duty_cycle : float, optional
            Fraction of time high (0–1).
        """
        f = SignalFunctions._resolve_frequency(frequency, period)

        def square_wave(tk: float) -> NDArray:
            return np.array(
                [amplitude * signal.square(2 * np.pi * f * tk, duty=duty_cycle)],
                dtype=float,
            )

        return square_wave
