from functools import wraps
import numpy as np
from numpy.typing import NDArray
from typing import (
    Callable,
    Set,
    Optional,
    get_type_hints,
    get_origin,
    get_args,
    Union,
    List,
)
import inspect
from decorators import Decorators


class Alias:
    def __init__(self):
        self._alias_for_x = {"x", "x_k", "state"}
        self._alias_for_u = {"u", "u_k", "input"}
        self._alias_for_t = {"t", "t_k", "time", "tau"}

    def get_alias_for_x(self) -> Set[str]:
        return self._alias_for_x

    def get_alias_for_u(self) -> Set[str]:
        return self._alias_for_u

    def get_alias_for_t(self) -> Set[str]:
        return self._alias_for_t

    @Decorators.verify_alias_input
    def set_alias_for_x(self, candidate_alias_set: Set[str]) -> None:
        self._alias_for_x = candidate_alias_set

    @Decorators.verify_alias_input
    def set_alias_for_u(self, candidate_alias_set: Set[str]) -> None:
        self._alias_for_u = candidate_alias_set

    @Decorators.verify_alias_input
    def set_alias_for_t(self, candidate_alias_set: Set[str]) -> None:
        self._alias_for_t = candidate_alias_set


class IOSafety(Alias):
    def __init__(self):
        super().__init__()

    @Decorators.verify_signature_and_parameter_names
    def set_f(self, f: Callable) -> Callable:
        if f is None:
            raise TypeError("The system function f cannot be None")
        return f

    @Decorators.verify_signature_and_parameter_names
    def set_h(self, h: Callable) -> Callable:
        if h is None:
            raise TypeError("The measurement function h cannot be None")
        return h

    @Decorators.verify_signature_and_parameter_names
    def set_u(self, u: Optional[Callable]) -> Callable:
        if u is None:
            u = lambda t: np.zeros((1, 1))
        return u

    @Decorators.verify_signature_and_parameter_names
    def set_Q(self, Q: Union[Callable, None], state_names: List[str]) -> Callable:
        if Q is None:

            def default_Q(x: NDArray, u: NDArray, t: float) -> NDArray:
                return np.zeros((len(state_names), len(state_names)))

            return default_Q
        return Q

    @Decorators.verify_signature_and_parameter_names
    def set_R(self, R: Union[Callable, None], measurement_names: List[str]) -> Callable:
        if R is None:

            def default_R(qx: NDArray, u: NDArray, t: float) -> NDArray:
                return np.zeros((len(measurement_names), len(measurement_names)))

            return default_R
        return R

    @Decorators.with_validated_args
    @Decorators.verify_signature_and_parameter_names
    def call_func(
        self,
        func: Callable,
        x: Optional[NDArray] = None,
        u: Optional[NDArray] = None,
        t: Optional[float] = None,
    ) -> NDArray:
        return func(
            x=x, u=u, t=t
        )  # with_validated_args will ensure this dispatches safely
