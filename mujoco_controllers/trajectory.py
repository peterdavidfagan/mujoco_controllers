"""Basic trajectory generation functions."""

import numpy as np

from dm_robotics.transformations import transformations as tr

class LinearTrajectory:
    """
    Generate a basic linear trajectory from current config to target config with osc.

    Note: this trajectory has no velocity or time parametrization and is for basic testing only.
    """

    def __init__(self, controller, target_pos, target_ori, num_points):
        self.controller = controller
        self.start_pos = controller.physics.bind(controller.eef_site).xpos.copy()
        self.end_pos = target_pos
        self.start_ori = controller.physics.bind(controller.eef_site).xmat.copy().reshape(3,3)
        self.start_ori = tr.mat_to_quat(self.start_ori)
        self.end_ori = target_ori
        self.num_points = num_points
        
        # linerly interpolate position
        self.positions = np.linspace(self.start_pos, self.end_pos, self.num_points)
        
        # slerp to interpolate orientation
        self.orientations = np.array([tr.quat_slerp(self.start_ori, self.end_ori, i/self.num_points) for i in range(self.num_points)])

    def execute_trajectory(self, duration):
        """Run the controller for a given duration."""
        step_duration = duration / self.num_points
        for position, orientation in zip(self.positions, self.orientations):
            self.controller.eef_target_position = position
            self.controller.eef_target_quat = orientation
            status = self.controller.run_controller(step_duration)
            if status:
                print("position error: ", self.controller.current_position_error())
                print("orientation error: ", self.controller.current_orientation_error())
                print("controller returned")
            else:
                raise RuntimeError("Controller failed to converge")
