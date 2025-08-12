import numpy as np
from numpy.typing import NDArray
from pykal_core.utils.signals import Generate, Transform
from pykal_core.utils.system import SafeIO
import inspect
from typing import (
    Any,
    Callable,
    Optional,
    get_type_hints,
    get_origin,
    Sequence,
    get_args,
    Union,
    List,
    Tuple,
)


class System:
    """
    Dynamical System

    """

    system_types = {"cti", "ctv", "dti", "dtv"}

    @staticmethod
    def f_zero(xk: NDArray) -> NDArray:
        return np.zeros_like(xk)

    @staticmethod
    def make_u_zero() -> Callable:
        def u(xk: NDArray) -> NDArray:
            return np.zeros_like(xk)

        return u

    @staticmethod
    def h_identity(xk: NDArray) -> NDArray:
        return xk

    @staticmethod
    def make_Q(
        *,
        state_names: List[str],
        multiply_eye_by_scalar: float = 0.01,
        create_Q_from_list_of_floats: Optional[List[float]] = None,
    ) -> Callable:
        def Q() -> NDArray:
            if create_Q_from_list_of_floats:
                Qmat = np.diag(create_Q_from_list_of_floats)
            else:
                Qmat = np.eye(len(state_names), len(state_names))
            return multiply_eye_by_scalar * Qmat

        return Q

    @staticmethod
    def make_R(
        *,
        measurement_names: List[str],
        multiply_eye_by_scalar: float = 0.01,
        create_R_from_list_of_floats: Optional[List[float]] = None,
    ) -> Callable:
        def R() -> NDArray:
            if create_R_from_list_of_floats:
                Rmat = np.diag(create_R_from_list_of_floats)
            else:
                Rmat = np.eye(len(measurement_names), len(measurement_names))
            return multiply_eye_by_scalar * Rmat

        return R

    def __init__(
        self,
        *,
        f: Optional[Callable] = None,
        h: Optional[Callable] = None,
        state_names: List[str],
        measurement_names: Optional[List[str]] = None,
        system_type: str = "cti",
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
    ) -> None:
        self.safeio = SafeIO(self)
        self._state_names = self.safeio._validate_string_sequence(state_names)
        self._measurement_names = (
            self.safeio._validate_string_sequence(measurement_names)
            if measurement_names is not None
            else [name + "_meas" for name in state_names]
        )

        self._system_type = self.safeio._validate_system_type(system_type)

        self._f = (
            self.safeio._validate_func_signature(f) if f is not None else System.f_zero
        )
        self._h = (
            self.safeio._validate_func_signature(h)
            if h is not None
            else System.h_identity
        )

        self._Q = (
            self.safeio._validate_func_signature(Q)
            if Q is not None
            else self.make_Q(state_names=state_names)
        )
        self._R = (
            self.safeio._validate_func_signature(R)
            if R is not None
            else self.make_R(measurement_names=self.measurement_names)
        )

    @property
    def system_type(self):
        return self._system_type

    @system_type.setter
    def system_type(self, system_type):
        self._system_type = self.safeio._validate_system_type(system_type)

    # Getters and Setters
    @property
    def state_names(self):
        return self._state_names

    @state_names.setter
    def state_names(self, names):
        self._state_names = self.safeio._validate_string_sequence(names)

    @property
    def measurement_names(self):
        return self._measurement_names

    @measurement_names.setter
    def measurement_names(self, names):
        self._measurement_names = self.safeio._validate_string_sequence(names)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        self._f = (
            self.safeio._validate_func_signature(f) if f is not None else System.f_zero
        )

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self._h = (
            self.safeio._validate_func_signature(h)
            if h is not None
            else System.h_identity
        )

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = (
            self.safeio._validate_func_signature(Q)
            if Q is not None
            else self.make_Q(state_names=self.state_names, multiply_eye_by_scalar=0)
        )

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = (
            self.safeio._validate_func_signature(R)
            if R is not None
            else self.make_R(
                measurement_names=self.measurement_names, multiply_eye_by_scalar=0
            )
        )


class Observer:
    def __init__(self, sys: System) -> None:
        self.sys = sys
        self.L = None  # add validation to these? Probably should at some point


class Controller:
    def __init__(self, sys: System):
        self.sys = sys
        self.C = None


class Signal:
    def __init__(self) -> None:
        self.generate = Generate()
        self.transform = Transform()
        self.G = None
