from pykal.control_system.system import System
from pykal.est.kf.ekf import EKF


class Observer:
    def __init__(self, system: System) -> None:

        self.sys = system
        self.ekf = EKF(system)
