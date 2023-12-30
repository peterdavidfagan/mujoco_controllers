"""
Differential Inverse Kinematics with various constraints.
"""

from dataclasses import dataclass

from jaxopt import OSQP
from mujoco_controllers import MujocoController


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
        self.eef_site = arm.attachment_site
        
        # variables used for computing control output
        self._eef_jacobian = None

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
        """Solve quadratic program to compute joint torques."""
        
        # define quadratic program Q, c
        self._compute_eef_jacobian()
        Q = self._eef_jacobian.T @ self._eef_jacobian
        c = -2 * self._eef_jacobian.T @ self.eef_target.velocity

        # define equality constraints A, b
        # for now I don't have any equality constraints

        # define inequality constraints G, h
        
        ## joint position limits
        pos_constraint_G = jnp.vstack([jnp.eye(self.physics.model.nv), -jnp.eye(self.physics.model.nv)])
        pos_constraint_G *= self.control_timestep # scale by control timestep to get more accurate limits
        pos_constraint_h = jnp.hstack([self.physics.model.jnt_range[:, 1], -self.physics.model.jnt_range[:, 0]])

        ## joint velocity limits
        vel_constraint_G = jnp.vstack([jnp.eye(self.physics.model.nv), -jnp.eye(self.physics.model.nv)])
        vel_constraint_h = jnp.hstack([self.physics.model.jnt_range[:, 1], -self.physics.model.jnt_range[:, 0]])
        
        ## combine all inequality constraints
        G = jnp.vstack([pos_constraint_G, vel_constraint_G])
        h = jnp.hstack([pos_constraint_h, vel_constraint_h])

        # run solver
        qp = OSQP()
        sol = qp.run(
            params_obj=(Q,c),
            params_ineq=(G,h),
            ).params
        
        print(sol.x)

        return sol
        

    def is_converged(self):
        pass
