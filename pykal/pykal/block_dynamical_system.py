import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Optional, Sequence
from pykal.safeio import SafeIO

class BlockDynamicalSystem:

    def __init__(
        self,
        *,
        f: Optional[Callable] = None,
        h: Optional[Callable] = None,
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
        sys_type: Optional[str] = None,
    ) -> None:

        # initialize state space and measurement space

        self._X = []
        self._Y = []

        # set dynamics function f and output map h
        def zero_dynamics(xk: NDArray) -> NDArray:
            return np.zeros_like(xk)

        self._f = f if f is not None else zero_dynamics

        def identity_map(xk: NDArray) -> NDArray:
            return xk

        self._h = h if h is not None else identity_map

        # set process noise and output noise matrix functions

        self._Q = Q if Q is not None else None
        self._R = R if R is not None else None

        self._sys_type = (
            sys_type if sys_type is not None else ("cti" if f is not None else None)
        )

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @X.setter
    def X(self,x):
        self._X = x

    @Y.setter
    def Y(self,y):
        self._Y = y

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        self._f = f

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self._h = h

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R

    @property
    def sys_type(self):
        return self._sys_type

    @sys_type.setter
    def sys_type(self, sys_type):
        self._sys_type = sys_type
