import inspect
from typing import Any, Callable, Dict, Optional


class DynamicalSystem:
    def __init__(
        self,
        *,
        f: Optional[Callable] = None,
        h: Callable,
    ) -> None:
        self._f = f
        self._h = h

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

    @staticmethod
    def _smart_call(
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

    def step(self, *, params: Dict[str, Any]) -> Any:
        if self.f is None:
            return self._smart_call(self.h, params)

        return self._smart_call(self.f, params), self._smart_call(self.h, params)
