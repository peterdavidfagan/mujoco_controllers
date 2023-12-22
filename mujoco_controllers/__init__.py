"""A very basic abstract base class for Mujoco Controller"""

import abc

class MujocoController(abc.ABC):
    """Abstract class for Mujoco Controller"""

    @abc.abstractmethod
    def compute_control_output(self):
        pass

    @abc.abstractmethod
    def is_converged(self):
        pass
