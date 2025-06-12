from enum import Enum, auto

class SystemType(Enum):
    r"""
    Enumeration of timing assumptions for plant / measurement models.

    Members
    -------
    CONTINUOUS_TIME_INVARIANT
        .. math::
           \dot x = f(x, u)
        (no explicit time argument)

    CONTINUOUS_TIME_VARYING
        .. math::
           \dot x = f(x, u, t)

    DISCRETE_TIME_INVARIANT
        .. math::
           x_{k+1} = f(x_k, u_k)

    DISCRETE_TIME_VARYING
        .. math::
           x_{k+1} = f(x_k, u_k, t_k)

    This enum exists purely for type-safety and clarity when specifying
    your system's timing assumptions.

    Examples
    --------
    >>> SystemType.CONTINUOUS_TIME_INVARIANT.name
    'CONTINUOUS_TIME_INVARIANT'
    >>> SystemType.DISCRETE_TIME_VARYING is SystemType['DISCRETE_TIME_VARYING']
    True
    """
    CONTINUOUS_TIME_INVARIANT = auto()
    CONTINUOUS_TIME_VARYING   = auto()
    DISCRETE_TIME_INVARIANT   = auto()
    DISCRETE_TIME_VARYING     = auto()
