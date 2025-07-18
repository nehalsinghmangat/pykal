from numpy.typing import NDArray
from pykal.control_system.system import System
import numpy as np
from typing import Callable, Optional, List


class Signals:
    def __init__(
        self, sys: System, user_defined_signal: Optional[Callable] = None
    ) -> None:
        self.sys = sys
        self.generate = Generate(sys)
        if user_defined_signal:
            self.user_signal = sys.safeio._validate_func_signature(user_defined_signal)


class Generate:
    def __init__(self, sys: System) -> None:
        self.sys = sys

    def make_constant_signal(self, output_list: List[float]) -> Callable:

        def constant_signal() -> NDArray:
            return np.array(output_list).reshape(-1, 1)

        return constant_signal

    def step(self, tk: float, func: Callable, step_time: float = 1) -> NDArray:
        if tk < step_time:
            return np.zeros((len(self.sys.state_names), 1))
        if tk >= step_time:
            return self.sys.safeio.smart_call(func, t=tk)
