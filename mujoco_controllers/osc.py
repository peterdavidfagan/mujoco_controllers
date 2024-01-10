"""
Operational Space Controller

Heavily inspired by Kevin Zakka's implementation: https://github.com/kevinzakka/mjctrl/blob/main/opspace.py
"""

from typing import Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

import mujoco
from mujoco import viewer

from dm_control import composer, mjcf

from mujoco_controllers.build_env import construct_physics

from hydra import compose, initialize
from hydra.utils import instantiate


@dataclass
class EEFTarget:
    position: np.ndarray
    velocity: np.ndarray
    quat: np.ndarray
    angular_velocity: np.ndarray

class OSC(object):

    def __init__(self, physics, arm, controller_config):
        # core simulation instance
        self.physics = physics
        
        # get site and joint details from arm
        self.arm = arm
        self.arm_joints = arm.joints
        self.arm_joint_ids = np.array(physics.bind(self.arm_joints).dofadr)
        self.eef_site = arm.attachment_site

        # controller parameters (timstepping, gains, convergence thresholds)
        self.control_steps = int(controller_config["control_dt"] // self.physics.model.opt.timestep)
        self.controller_gains = controller_config["gains"]
        self.position_threshold = controller_config["convergence"]["position_threshold"]
        self.orientation_threshold = controller_config["convergence"]["orientation_threshold"]
        self.nullspace_config = np.array(controller_config["nullspace"]["joint_config"]) # rename to manipulability config
        
        # relevant control variables (TODO: merge into one call due to dependency)
        self._eef_jacobian = None
        self._eef_mass_matrix = None
        
        # end effector targets
        self.eef_target = EEFTarget(
            position=None,
            velocity=None,
            quat=None,
            angular_velocity=None,
        )
    
    @property
    def current_eef_position(self):
        return self.physics.bind(self.eef_site).xpos.copy()

    @property
    def current_eef_quat(self):
        quat = np.zeros(4,)
        rot_mat = self.physics.bind(self.eef_site).xmat.copy()
        mujoco.mju_mat2Quat(quat, rot_mat)
        # TODO: check if this is necessary the quaternion may already be normalized from mujoco
        # ensure unit quaternion
        quat /= np.linalg.norm(quat)
        return quat

    @property
    def current_eef_velocity(self):
        return self._eef_jacobian[:3,:] @ self.physics.data.qvel[self.arm_joint_ids]

    @property
    def current_eef_angular_velocity(self):
        return self._eef_jacobian[3:,:] @ self.physics.data.qvel[self.arm_joint_ids]

    def set_target(self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        quat: np.ndarray = None,
        angular_velocity: np.ndarray = None,
    ):
        self.eef_target = EEFTarget(
            position=position if position is not None else self.eef_target.position,
            velocity=velocity if velocity is not None else self.eef_target.velocity,
            quat=quat/np.linalg.norm(quat) if quat is not None else self.eef_target.quat,
            angular_velocity=angular_velocity if angular_velocity is not None else self.eef_target.angular_velocity,
        )

    def _compute_eef_mass_matrix(self):
        nv = self.physics.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.physics.model.ptr, M, self.physics.data.qM)
        M = M[self.arm_joint_ids, :][:, self.arm_joint_ids] # filter for links we care about
        self.arm_mass_matrix = M

        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(self._eef_jacobian, np.dot(M_inv, self._eef_jacobian.T))
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            # do the linalg inverse if matrix is non-singular
            # because it's faster and more accurate
            self._eef_mass_matrix = np.linalg.inv(Mx_inv)
        else:
            # using the rcond to set singular values < thresh to 0
            # singular values < (rcond * max(singular_values)) set to 0
            self._eef_mass_matrix = np.linalg.pinv(Mx_inv, rcond=1e-2)

    def _compute_eef_jacobian(self):
        nv = self.physics.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(
                m=self.physics.model.ptr,
                d=self.physics.data.ptr, 
                jacp=jacp, 
                jacr=jacr, 
                site=self.physics.bind(self.eef_site).element_id
                )
        jacp = jacp[:, self.arm_joint_ids] # filter jacobian for joints we care about
        jacr = jacr[:, self.arm_joint_ids] # filter jacobian for joints we care about
        self._eef_jacobian = np.vstack([jacp, jacr])
    
    def _orientation_error(
        self,
        quat: np.ndarray,
        quat_des: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the orientation error between two quaternions.

        Implemented base on: Resolved-acceleration control of robot manipulators: A critical review with experiments.
        Assumed that the quaternions are unit quaternions.
        """
        quat_conj = np.zeros(4,)
        mujoco.mju_negQuat(quat_conj, quat)
        quat_conj /= np.linalg.norm(quat_conj)

        quat_err = np.zeros(4,)
        mujoco.mju_mulQuat(quat_err, quat_des, quat_conj)
        
        return quat_err[1:] * np.sign(quat_err[0])

    # TODO: move to properties
    def current_orientation_error(self):
        return np.max(abs(self._orientation_error(self.current_eef_quat, self.eef_target.quat)))

    def current_position_error(self):
        return  np.linalg.norm(self.current_eef_position - self.eef_target.position)

    # TODO: refactor so no conditionals in this function
    def pd_control(
        self,
        x: np.array,
        x_desired: np.array,
        dx: np.array,
        dx_desired: np.array,
        gains:Tuple,
        mode="position"):
    
        if mode == "position":
            pos_error = np.clip(x_desired - x, -0.05, 0.05) # clip to prevent large errors (this is suboptimal)
            vel_error = dx_desired - dx
            return gains["kp"] * pos_error + gains["kd"] * vel_error
        elif mode == "orientation": 
            return gains["kp"] * self._orientation_error(x, x_desired) + gains["kd"] * (dx_desired - dx)
        elif mode == "nullspace":
            return gains["kp"] * (x_desired - x) + gains["kd"] * (dx_desired - dx)
        else:
            raise ValueError("Invalid mode for pd control")
    
    def compute_control_output(self):
        """ Compute the control output for the robot arm. """
        # update control member variables
        self._compute_eef_jacobian()
        self._compute_eef_mass_matrix()
        
        # calculate position pd control
        position_pd = self.pd_control(
            x=self.current_eef_position,
            x_desired=self.eef_target.position,
            dx=self.current_eef_velocity,
            dx_desired=self.eef_target.velocity,
            gains=self.controller_gains["position"],
            mode="position"
        )
        
        # calculate orientation pd control
        orientation_pd = self.pd_control(
            x=self.current_eef_quat,
            x_desired=self.eef_target.quat,
            dx=self.current_eef_angular_velocity,
            dx_desired=self.eef_target.angular_velocity,
            gains=self.controller_gains["orientation"],
            mode="orientation"
                )
        
        pd_error = np.hstack([position_pd, orientation_pd])
        tau = self._eef_jacobian.T @ self._eef_mass_matrix @ pd_error 
        
        # secondary task of moving towards configuration with high manipulability
        nullspace_error = self.pd_control(
                x=self.physics.data.qpos[self.arm_joint_ids],
                x_desired=self.nullspace_config,
                dx=self.physics.data.qvel[self.arm_joint_ids],
                dx_desired=np.zeros(len(self.arm_joints)),
                gains=self.controller_gains["nullspace"],
                mode="nullspace"
                )
        null_jacobian = np.linalg.inv(self.arm_mass_matrix) @ self._eef_jacobian.T @ self._eef_mass_matrix
        tau += (np.eye(len(self.arm_joints)) - self._eef_jacobian.T @ null_jacobian.T) @ nullspace_error
       
        # compensate for external forces (gravity, coriolis)
        tau += self.physics.data.qfrc_bias[self.arm_joint_ids]

        # compute effective torque
        actuator_moment_inv = np.linalg.pinv(self.physics.data.actuator_moment)
        actuator_moment_inv = actuator_moment_inv[self.arm_joint_ids, :][:, self.arm_joint_ids]
        tau = tau @ actuator_moment_inv 

        return tau

    def is_converged(self):
        """ Check if the robot arm is converged to the target. """
        # TODO: add check for velocities
        if (self.current_position_error() < self.position_threshold) and \
                (self.current_orientation_error() < self.orientation_threshold):
            return True
        else:
            return False


if __name__ == "__main__":
        
    # save default configs (TODO: move to keyframe)
    ready_config = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    # load different robot configurations
    initialize(version_base=None, config_path="./config", job_name="default_config")
    cfg = compose(config_name="itl_rearrangement")
    
    # ensure mjcf paths are relative to this file
    file_path = Path(__file__).parent.absolute()
    cfg.robots.arm.arm.mjcf_path = str(file_path / cfg.robots.arm.arm.mjcf_path)
    cfg.robots.end_effector.end_effector.mjcf_path = str(file_path / cfg.robots.end_effector.end_effector.mjcf_path)

    # instantiate physics and controller
    physics, passive_view, arm, gripper = construct_physics(cfg)
    osc = arm.controller_config.controller(physics, arm) # super unclean fix this
    control_steps = int(arm.controller_config.controller_params.control_dt // physics.model.opt.timestep)

    # run controller to random targets
    for _ in range(10):
        # randomly samply target pose from workspace
        position_x = np.random.uniform(0.25, 0.5)
        position_y = np.random.uniform(-0.3, 0.3)
        position_z = np.random.uniform(0.6, 0.7)
        position = np.array([position_x, position_y, position_z])
        
        # fix gripper orientation
        quat = np.zeros(4,)
        mat = obj_rot = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix().flatten()
        mujoco.mju_mat2Quat(quat, mat)

        osc.set_target(
            position=position,
            velocity=np.zeros(3),
            quat=quat,
            angular_velocity=np.zeros(3)
        )

        duration = 5
        converged = False
        start_time = physics.data.time
        while (physics.data.time - start_time < duration) and (not converged):
            # compute control command
            arm_command = osc.compute_control_output()
            gripper_command = np.array([0.0])
            control_command = np.hstack([arm_command, gripper_command])

            # step the simulation
            for _ in range(control_steps):
                physics.set_control(control_command)
                physics.step()
                if passive_view is not None:
                    passive_view.sync()

                if osc.is_converged():
                    converged = True
                    break

