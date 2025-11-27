import inspect
from typing import Any, Callable, Dict, Optional

class DynamicalSystem:
    def __init__(
        self,
        *,
        f: Optional[Callable] = None,
        h: Optional[Callable] = None,
        state_name: Optional[str] = None,
    ) -> None:
        self._f = f
        self._h = h if h is not None else (lambda state_vector: state_vector)

        if f is not None and state_name is None:
            raise ValueError("state_name cannot be 'None' if f is not 'None'")

        self._state_name = state_name

    @property
    def f(self) -> Optional[Callable]:
        return self._f

    @f.setter
    def f(self, f: Optional[Callable]) -> None:
        self._f = f

    @property
    def h(self) -> Callable:
        return self._h

    @h.setter
    def h(self, h: Callable) -> None:
        self._h = h

    @property
    def state_name(self) -> Optional[str]:
        return self._state_name

    @state_name.setter
    def state_name(self, state_name: Optional[str]) -> None:
        self._state_name = state_name

    def smart_call(
        self,
        func: Callable[..., Any],
        param_dict: Dict[str, Any],
    ) -> Any:
        sig = inspect.signature(func)
        params = sig.parameters
        pool: Dict[str, Any] = dict(param_dict) if param_dict else {}

        pos_args = []
        for name, p in params.items():
            if p.kind is inspect.Parameter.POSITIONAL_ONLY and name in pool:
                pos_args.append(pool.pop(name))

        call_kwargs: Dict[str, Any] = {}
        for name, p in params.items():
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                if name in pool:
                    call_kwargs[name] = pool.pop(name)

        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            call_kwargs.update(pool)

        bound = sig.bind_partial(*pos_args, **call_kwargs)
        return func(*bound.args, **bound.kwargs)

    def step(
        self,
        *,
        param_dict: Dict[str, Any],
    ) -> Any:
        if self.f is None:
            return self.smart_call(self.h, param_dict)

        param_dict = dict(param_dict) # copy param_dict so we dont risk mutation outside of this method
        param_dict[self.state_name] = self.smart_call(self.f, param_dict) 
        return self.smart_call(self.h, param_dict)
