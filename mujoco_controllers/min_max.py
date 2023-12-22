""" A ridicuously simple end effector min/max controller. """
from mujoco_controllers import MujocoController


# this is a bit of a crappy controller lol

class MinMax(MujocoController):
    def __init__(self, max_val, min_val):
        """ Initialize the controller. """
        self.max_val = max_val
        self.min_val = min_val
        self._status = "min"

    def compute_control_output(self):
        """ Compute the control signal. """
        if self._status == 'max':
            return self.max_val
        else:
            return self.min_val

    @property
    def status(self):
        """ Return the status of the controller. """
        return self._status

    @status.setter
    def status(self, new_status):
        """ Set the status of the controller. """
        assert new_status in ['min', 'max']
        self._status = new_status

    def is_converged(self):
        pass
