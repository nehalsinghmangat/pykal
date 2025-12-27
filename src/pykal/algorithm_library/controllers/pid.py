import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class PID:
    @staticmethod
    def f(
        ck: Tuple[float, float, float], rk: float, xhat_k: float
    ) -> Tuple[float, float, float]:
        """
        Perform one state update step for a discrete-time PID controller.

        Parameters
        ----------
        ck : Tuple[float, float, float]
            A tuple ``(ek_prev, Ik_prev, ek_prev_prev)`` containing:
                - ``ek_prev``      : the current error from the previous iteration
                - ``Ik_prev``      : the accumulated integral of error
                - ``ek_prev_prev`` : the error from two iterations ago (unused)

        rk : float
            The reference (setpoint) value at time k.

        xhat_k : float
            The current state estimate (measured or estimated output) at time k.

        Returns
        -------
        (ek, Ik, ek_prev) : Tuple[float, float, float]
            The updated controller state:
                - ``ek``      : current error at time k
                - ``Ik``      : updated integral of error
                - ``ek_prev`` : previous error (for derivative calculation)

        Notes
        -----
        This function updates the internal state of the PID controller following
        the discrete-time formulation:

        **Error calculation:**
            ``ek = rk - xhat_k``

        **Integral update:**
            ``Ik = Ik_prev + ek``

        **State propagation:**
            The current error becomes the previous error for the next iteration,
            enabling the derivative term calculation in ``standard_h``.

        The state tuple always maintains the structure:
            ``(current_error, integral, previous_error)``
        """
        ek_prev, Ik_prev, _ = ck
        ek = rk - xhat_k
        Ik = Ik_prev + ek
        return (ek, Ik, ek_prev)

    @staticmethod
    def h(
        ck: Tuple[float, float, float],
        rk: float,
        xhat_k: float,
        KP: float,
        KI: float,
        KD: float,
    ) -> float:
        """
        Compute the PID control output from the current controller state.

        Parameters
        ----------
        ck : Tuple[float, float, float]
            A tuple ``(ek, Ik, ek_prev)`` containing:
                - ``ek``      : current error at time k
                - ``Ik``      : accumulated integral of error
                - ``ek_prev`` : previous error for derivative calculation

        rk : float
            The reference (setpoint) value at time k (unused in output calculation).

        xhat_k : float
            The current state estimate at time k (unused in output calculation).

        KP : float
            Proportional gain coefficient.

        KI : float
            Integral gain coefficient.

        KD : float
            Derivative gain coefficient.

        Returns
        -------
        uk : float
            The control output signal at time k.

        Notes
        -----
        This function computes the standard discrete-time PID control law:

        **PID control equation:**
            ``uk = KP * ek + KI * Ik + KD * (ek - ek_prev)``

        where:
            - **Proportional term:** ``KP * ek`` provides immediate response to current error
            - **Integral term:** ``KI * Ik`` eliminates steady-state error
            - **Derivative term:** ``KD * (ek - ek_prev)`` provides damping and anticipates future error

        The parameters ``rk`` and ``xhat_k`` are included in the signature for
        compatibility with the DynamicalSystem framework but are not used in the
        computation, as the error is already available in the state tuple ``ck``.
        """
        ek, Ik, ek_prev = ck
        uk = KP * ek + KI * Ik + KD * (ek - ek_prev)
        return uk


# Module-level aliases for convenience
# Allows usage like: from pykal.algorithm_library.controllers import pid; pid.f(...)
f = PID.f
h = PID.h
