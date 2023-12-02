"""Franka Emika Panda Robot Arm."""

import numpy as np

from dm_control import mjcf
from dm_robotics.moma.models.robots.robot_arms import robot_arm


class FER(robot_arm.RobotArm):
    """Franka Emika Panda Robot Arm."""

    def __init__(self, relative_robot_mjcf_path: str, actuator_config: dict, sensor_config: dict=None):
        """Initialize the robot arm."""
        self.relative_robot_mjcf_path = relative_robot_mjcf_path
        self.actuator_config = actuator_config
        self.sensor_config = sensor_config
        super().__init__()

    def _build(self):
        self._fer_root = mjcf.from_path(self.relative_robot_mjcf_path)
        self._joints = self._fer_root.find_all("joint")
        # TODO: add assert that checks for joint actuator assignment
        self._add_actuators()
        self._add_sensors()
        self._actuators = self._fer_root.find_all("actuator")
        #self._wrist_site = self._fer_root.find("site", "wrist_site")
        self._attachment_site = self._fer_root.find("site", "attachment_site")
        self._wrist_site = self._attachment_site

    def _add_actuators(self):
        """Override the actuator model by config."""
        if self.actuator_config["type"] == "motor":
            for idx, (joint, joint_type) in enumerate(self.actuator_config["joint_actuator_mapping"].items()):
                print("Adding actuator for joint: {}".format(joint))
                actuator = self._fer_root.actuator.add(
                    "motor",
                    name="actuator{}".format(idx + 1),
                    **self.actuator_config[joint_type],
                )
                actuator.joint = self._joints[idx]

        elif self.actuator_config["type"] == "general":
            for idx, (joint, joint_type) in enumerate(self.actuator_config["joint_actuator_mapping"].items()):
                print("Adding actuator for joint: {}".format(joint))
                actuator = self._fer_root.actuator.add(
                    "general",
                    name="actuator{}".format(idx + 1),
                    **self.actuator_config[joint_type],
                )
                actuator.joint = self._joints[idx]
        
        elif self.actuator_config["type"] == "velocity":
            for idx, (joint, joint_type) in enumerate(self.actuator_config["joint_actuator_mapping"].items()):
                print("Adding actuator for joint: {}".format(joint))
                actuator = self._fer_root.actuator.add(
                    "velocity",
                    name="actuator{}".format(idx + 1),
                    **self.actuator_config[joint_type],
                )
                actuator.joint = self._joints[idx]

        else:
            raise ValueError("Unsupported actuator model: {}".format(self.actuator_model))

    def _add_sensors(self):
        """Override the sensor model by config."""
        if self.sensor_config["type"] == "jointpos":
            for idx, (joint, joint_type) in enumerate(self.sensor_config["joint_sensor_mapping"].items()):
                print("Adding sensor for joint: {}".format(joint))
                sensor = self._fer_root.sensor.add(
                    "jointpos",
                    **self.sensor_config[joint],
                )
                sensor.joint = self._joints[idx]

    @property
    def joints(self):
        """Returns a list of joints in the robot."""
        return self._joints

    @property
    def actuators(self):
        """Returns a list of actuators in the robot."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the MJCF model for the robot."""
        return self._fer_root

    @property
    def name(self):
        """Returns the name of the robot."""
        return "franka_emika_panda"

    @property
    def wrist_site(self):
        """Returns the wrist site."""
        return self._wrist_site

    @property
    def attachment_site(self):
        """Returns the attachment site."""
        return self._attachment_site

    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        """Set the joint angles of the robot."""
        physics.bind(self._joints).qpos = qpos
