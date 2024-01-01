"""
Differential Inverse Kinematics with various constraints.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

import mujoco
from mujoco import viewer
from mujoco_controllers import MujocoController
from mujoco_controllers.build_env import construct_physics

from jaxopt import OSQP

from hydra import compose, initialize
from hydra.utils import instantiate

@dataclass
class EEFTarget:
    position: np.ndarray
    velocity: np.ndarray
    quat: np.ndarray
    angular_velocity: np.ndarray

class DiffIK(MujocoController):

    def __init__(self, physics, arm, controller_config):
        
        # core simulation instance 
        self.physics = physics
                
        # get site and joint details from arm
        self.arm = arm
        self.arm_joints = arm.joints
        self.arm_joint_ids = np.array(physics.bind(self.arm_joints).dofadr)
        self.num_arm_joints = len(self.arm_joints)
        self.eef_site = arm.attachment_site
        
        # controller params
        self.position_threshold = controller_config["convergence"]["position_threshold"]
        self.orientation_threshold = controller_config["convergence"]["orientation_threshold"]


        # variables used for computing control output
        self._eef_jacobian = None
        self.control_timestep = controller_config["control_dt"]

        # initialise without end effector targets
        self.eef_target = EEFTarget(
            position=None,
            velocity=None,
            quat=None,
            angular_velocity=None,
        )

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
        
        return quat_err

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
    
    def compute_control_output(self):
        """Solve quadratic program to compute joint velocities."""
       
        # compute desired twist from current eef state to target eef state
        # in one control cycle (TODO: make this more stable)
        twist = np.zeros(6)
        twist[:3] = (self.eef_target.position - self.current_eef_position) / self.control_timestep
         
        vel = np.zeros(3,)
        quat_err = self._orientation_error(
            self.current_eef_quat,
            self.eef_target.quat,
        )
        mujoco.mju_quat2Vel(vel, quat_err, self.control_timestep)
        twist[3:] = vel

        # TODO: add scaling factor that gets tuned
        #twist *= 0.05

        # define quadratic program Q, c
        self._compute_eef_jacobian()
        Q = self._eef_jacobian.T @ self._eef_jacobian
        c = -self._eef_jacobian.T @ twist

        # define equality constraints A, b
        # for now I don't have any equality constraints
        
        # define inequality constraints G, h
        ## joint position limits
        #pos_constraint_G = jnp.vstack([jnp.eye(self.num_arm_joints), -jnp.eye(self.num_arm_joints)])
        #pos_constraint_G *= self.control_timestep # scale by control timestep to get more accurate limits
        #pos_constraint_h = jnp.hstack([self.physics.model.jnt_range[self.arm_joint_ids, 1], self.physics.model.jnt_range[self.arm_joint_ids, 0]])

        ## joint velocity limits
        vel_constraint_G = jnp.vstack([jnp.eye(self.num_arm_joints), -jnp.eye(self.num_arm_joints)])
        vel_constraint_h = jnp.hstack([self.physics.model.actuator_ctrlrange[:7, 1], -self.physics.model.actuator_ctrlrange[:7, 0]])
        
        ## combine all inequality constraints
        G = vel_constraint_G
        h = vel_constraint_h
        #G = jnp.vstack([pos_constraint_G, vel_constraint_G])
        #h = jnp.hstack([pos_constraint_h, vel_constraint_h])
        
        # run solver
        qp = OSQP()
        sol = qp.run(
            params_obj=(Q,c),
            params_ineq=(G,h),
            ).params
        

        return sol.primal
    

    # TODO: move to properties
    def current_orientation_error(self):
        quat_err = self._orientation_error(self.current_eef_quat, self.eef_target.quat)
        return np.max(abs(np.sign(quat_err[0]) * quat_err[1:]))

    def current_position_error(self):
        return  np.linalg.norm(self.current_eef_position - self.eef_target.position)
       
    def is_converged(self):
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
    cfg = compose(
            config_name="itl_rearrangement", 
            overrides=[
                "control_dt=0.1", # frequency of diffik controller
                "robots/arm/controller_config=diffik", 
                "robots/arm/actuator_config=velocity",
                ]
            )

    # ensure mjcf paths are relative to this file (TODO: make this cleaner)
    file_path = Path(__file__).parent.absolute()
    cfg.robots.arm.arm.mjcf_path = str(file_path / cfg.robots.arm.arm.mjcf_path)
    cfg.robots.end_effector.end_effector.mjcf_path = str(file_path / cfg.robots.end_effector.end_effector.mjcf_path)

    # instantiate physics and controller
    physics, passive_view, arm, gripper = construct_physics(cfg)
    diffik = arm.controller_config.controller(physics, arm) # super unclean fix this
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

        diffik.set_target(
            position=position,
            velocity=np.zeros(3),
            quat=quat,
            angular_velocity=np.zeros(3)
        )

        duration = 5
        converged = False
        start_time = physics.data.time
        while (not converged): # ignore duration while debugging
            # compute control command
            arm_command = diffik.compute_control_output()
            gripper_command = np.array([0.0])
            control_command = np.hstack([arm_command, gripper_command])

            # step the simulation
            for _ in range(control_steps):
                physics.set_control(control_command)
                physics.step()
                if passive_view is not None:
                    passive_view.sync()

                if diffik.is_converged():
                    converged = True
                    break
