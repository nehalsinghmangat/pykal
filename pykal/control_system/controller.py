from typing import (
    Callable,
    Optional,
)
from pykal.control_system.system import System
from pykal.signals.signals import Signals


class Controller:
    def __init__(self, sys: System, u: Optional[Callable] = None):
        self.sys = sys
        self.signals = Signals(sys)
        self._u = (
            sys.make_u_zero() if u is None else sys.safeio._validate_func_signature(u)
        )

    @property
    def u(self) -> Optional[Callable]:
        return self._u

    @u.setter
    def u(self, func: Callable) -> None:
        self._u = self.sys.safeio._validate_func_signature(func)
