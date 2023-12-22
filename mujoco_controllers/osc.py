"""
Operational Space Controller

Heavily inspired by Kevin Zakka's implementation: https://github.com/kevinzakka/mjctrl/blob/main/opspace.py
"""

from typing import Tuple, Dict, Optional, Union, List

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

import mujoco
from mujoco import viewer

from dm_control import composer, mjcf
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma import robot
from dm_robotics.transformations.transformations import mat_to_quat, quat_to_mat, quat_to_euler
from dm_robotics.transformations import transformations as tr

from mujoco_controllers.build_env import construct_physics

from rearrangement_benchmark.environment.props import Rectangle
from ikpy.chain import Chain
from hydra import compose, initialize
from hydra.utils import instantiate

class OSC(object):

    def __init__(self, physics, arm, controller_config):
        # core simulation instance
        self.physics = physics

        # controller gains
        self.controller_timestep = controller_config["control_dt"]
        self.control_steps = int(self.controller_timestep // self.physics.model.opt.timestep)
        self.controller_gains = controller_config["gains"]
        self.nullspace_config = np.array(controller_config["nullspace"]["joint_config"])
        self.position_threshold = controller_config["convergence"]["position_threshold"]
        self.orientation_threshold = controller_config["convergence"]["orientation_threshold"]
        
        # get site and joint details from arm
        self.arm = arm
        self.eef_site = arm.attachment_site
        self.arm_joints = arm.joints
        self.arm_joint_ids = np.array(physics.bind(self.arm_joints).dofadr)

        # control targets
        self._eef_target_position = None
        self._eef_target_velocity = None
        self._eef_target_quat = None
        self._eef_target_angular_velocity = None
        
        # control equation variables
        self._eef_mass_matrix = None
        self._eef_jacobian = None

    @property
    def eef_target_position(self):
        return self._eef_target_position
    
    @eef_target_position.setter
    def eef_target_position(self, value):
        self._eef_target_position = value

    @property
    def eef_target_quat(self):
        return self._eef_target_quat

    @eef_target_quat.setter
    def eef_target_quat(self, value):
        self._eef_target_quat = value
    
    @property
    def eef_target_velocity(self):
        return self._eef_target_velocity

    @eef_target_velocity.setter
    def eef_target_velocity(self, value):
        self._eef_target_velocity = value

    @property
    def eef_target_angular_velocity(self):
        return self._eef_target_angular_velocity

    @eef_target_angular_velocity.setter
    def eef_target_angular_velocity(self, value):
        self._eef_target_angular_velocity = value
    
    @property
    def current_eef_position(self):
        return self.physics.bind(self.eef_site).xpos

    @property
    def current_eef_quat(self):
        return mat_to_quat(self.physics.bind(self.eef_site).xmat.reshape(3,3))

    def _compute_eef_mass_matrix(self):
        nv = self.physics.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.physics.model.ptr, M, self.physics.data.qM)
        M = M[self.arm_joint_ids, :][:, self.arm_joint_ids]
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
        mujoco.mj_jacSite(m=self.physics.model.ptr, d=self.physics.data.ptr, jacp=jacp, jacr=jacr, site=self.physics.bind(self.eef_site).element_id)
        jacp = jacp[:, self.arm_joint_ids] # filter jacobian for joints we care about
        jacr = jacr[:, self.arm_joint_ids] # filter jacobian for joints we care about
        self._eef_jacobian = np.vstack([jacp, jacr])
    
    def _orientation_error(
        self,
        quat: np.ndarray,
        quat_des: np.ndarray,
    ) -> np.ndarray:
        quat_err = tr.quat_mul(quat, tr.quat_conj(quat_des))
        quat_err /= np.linalg.norm(quat_err)
        axis_angle = tr.quat_to_axisangle(quat_err)
        if quat_err[0] < 0.0:
            angle = np.linalg.norm(axis_angle) - 2 * np.pi
        else:
            angle = np.linalg.norm(axis_angle)
        return axis_angle * angle

    def _calc_damping(self, gains: Dict[str, float]) -> np.ndarray:
        return gains["damping_ratio"] * 2 * np.sqrt(gains["kp"])

    def current_orientation_error(self):
        eef_orientation = self.physics.bind(self.eef_site).xmat.reshape(3, 3)
        eef_orientation = tr.mat_to_quat(eef_orientation)
        return np.max(abs(self._orientation_error(eef_orientation, self.eef_target_quat)))

    def current_position_error(self):
        eef_position = self.physics.bind(self.eef_site).xpos
        return  np.linalg.norm(eef_position - self.eef_target_position)

    def pd_control(
        self,
        x: np.array,
        x_desired: np.array,
        dx: np.array,
        dx_desired: np.array,
        gains:Tuple,
        mode="position"):
    
        if mode == "position":
            try:
                gains = self.controller_gains["position"]
            except:
                raise ValueError("Invalid controller gains")
            
            pos_error = np.clip(x_desired - x, -0.05, 0.05)
            vel_error = dx_desired - dx
            if gains["kd"] is None:
                error = gains["kp"] * pos_error + self._calc_damping(gains)  * vel_error
            else:
                error = gains["kp"] * pos_error + gains["kd"] * vel_error
            # considered limiting error term
            return error
            
        elif mode == "orientation": 
            try:
                gains = self.controller_gains["orientation"]
            except:
                raise ValueError("Invalid controller gains")
            
            # negative sign for kp arises due to how orientation error is currently calculated
            if gains["kd"] is None:
                error = -gains["kp"] * self._orientation_error(x, x_desired) + self._calc_damping(gains) * (dx_desired - dx)
            else:
                error = -gains["kp"] * self._orientation_error(x, x_desired) + gains["kd"] * (dx_desired - dx)
            return error
        
        elif mode == "nullspace":
            try:
                gains = self.controller_gains["nullspace"]
            except:
                raise ValueError("Invalid controller gains")
            
            if gains["kd"] is None:
                error = gains["kp"] * (x_desired - x) + self._calc_damping(gains) * (dx_desired - dx)
            else:
                error = gains["kp"] * (x_desired - x) + gains["kd"] * (dx_desired - dx)
            return error
        
        else:
            raise ValueError("Invalid mode for pd control")
    
    def compute_control_output(self):
        """ Compute the control output for the robot arm. """
        # update control member variables
        self._compute_eef_jacobian()
        self._compute_eef_mass_matrix()

        # get joint velocities
        current_joint_velocity = self.physics.data.qvel[self.arm_joint_ids]
        
        # calculate position pd control
        eef_current_position = self.physics.bind(self.eef_site).xpos.copy()
        eef_current_velocity = self._eef_jacobian[:3,:] @ current_joint_velocity
        position_pd = self.pd_control(
            x=eef_current_position,
            x_desired=self._eef_target_position,
            dx=eef_current_velocity,
            dx_desired=self._eef_target_velocity,
            gains=self.controller_gains["position"],
            mode="position"
        )
        
        # calculate orientation pd control
        eef_quat = mat_to_quat(self.physics.bind(self.eef_site).xmat.reshape(3,3).copy())
        eef_angular_vel = self._eef_jacobian[3:,:] @ current_joint_velocity
        orientation_pd = self.pd_control(
            x=eef_quat,
            x_desired=self._eef_target_quat,
            dx=eef_angular_vel,
            dx_desired=self._eef_target_angular_velocity,
            gains=self.controller_gains["orientation"],
            mode="orientation"
                )
        
        pd_error = np.hstack([position_pd, orientation_pd])
        tau = self._eef_jacobian.T @ self._eef_mass_matrix @ pd_error 
        
        # nullspace projection
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
       
        # compensate for external forces
        tau += self.physics.data.qfrc_bias[self.arm_joint_ids]

        # compute effective torque
        actuator_moment_inv = np.linalg.pinv(self.physics.data.actuator_moment)
        actuator_moment_inv = actuator_moment_inv[self.arm_joint_ids, :][:, self.arm_joint_ids]
        tau = tau @ actuator_moment_inv 

        return tau

    def is_converged(self):
        """ Check if the robot arm is converged to the target. """
        arm_converged = False

        # check for arm controller convergence
        eef_pos = self.physics.bind(self.eef_site).xpos.copy()
        eef_quat = mat_to_quat(self.physics.bind(self.eef_site).xmat.reshape(3,3).copy())
        eef_vel = self._eef_jacobian[:3,:] @ self.physics.data.qvel[self.arm_joint_ids]
        eef_angular_vel = self._eef_jacobian[3:,:] @ self.physics.data.qvel[self.arm_joint_ids]
        
        if (self.current_position_error() < self.position_threshold) and (self.current_orientation_error() < self.orientation_threshold):
            arm_converged = True

        return arm_converged


if __name__ == "__main__":
    
    # save default configs 
    ready_config = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    # load different robot configurations
    initialize(version_base=None, config_path="./config", job_name="default_config")
    POSITION_CONFIG = compose(config_name="controller_tuning", overrides=["robots=default"])
    VELOCITY_CONFIG = compose(config_name="controller_tuning", overrides=["robots=velocity"])
    MOTOR_CONFIG = compose(config_name="controller_tuning", overrides=["robots=motor"])

    # For now assign default cfg
    cfg = MOTOR_CONFIG
    physics, passive_view, arm, gripper = construct_physics(cfg)
    osc = OSC(physics, arm, gripper, MOTOR_CONFIG["controllers"]["osc"], passive_view)
   

    # get object pose
    cube_id = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, "cube/cube")
    object_pos = physics.data.geom_xpos[cube_id]
    object_orientation = physics.data.geom_xmat[cube_id].reshape(3,3)

    print("Object position: ", object_pos)
    print("Object orientation: ", object_orientation)
    
    pre_pick_height = 0.6
    pick_height = 0.15
    default_quat =  mat_to_quat(R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix())


    # prepick
    osc.eef_target_position = object_pos + np.array([0, 0, pre_pick_height])
    osc.eef_target_quat = default_quat
    osc.eef_target_velocity = np.zeros(3)
    osc.eef_target_angular_velocity = np.zeros(3)
    osc.run_controller(2.0)

    # pick
    osc.eef_target_position = object_pos + np.array([0, 0, pick_height])
    osc.run_controller(2.0)

    # close gripper
    osc._gripper_status = "closing"
    osc.run_controller(1.0)

    # lift
    osc.eef_target_position = object_pos + np.array([0, 0, pre_pick_height])
    osc.run_controller(2.0)

