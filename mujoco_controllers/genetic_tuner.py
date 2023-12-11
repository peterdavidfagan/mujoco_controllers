"""Tuning control params with genetic algorithm."""

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
from mujoco_controllers.osc import OSC

from rearrangement_benchmark.env_components.props import Rectangle
from ikpy.chain import Chain
from hydra import compose, initialize
from hydra.utils import instantiate

import numpy as np
import jax
from evosax import CMA_ES


if __name__=="__main__":
    # save default configs 
    ready_config = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    grasp_pose_config = np.array([-3.95380744e-04,  2.37985323e-01,  3.52180384e-04, -2.55912981e+00,
     -2.42755642e-04,  2.79711454e+00,  7.85573570e-01])

    # load different robot configurations
    initialize(version_base=None, config_path="./config", job_name="default_config")
    POSITION_CONFIG = compose(config_name="controller_tuning", overrides=["robots=default"])
    VELOCITY_CONFIG = compose(config_name="controller_tuning", overrides=["robots=velocity"])
    MOTOR_CONFIG = compose(config_name="controller_tuning", overrides=["robots=motor"])
    IKPY_URDF_PATH = "./models/arms/robot.urdf"

    # for now assign default cfg
    cfg = MOTOR_CONFIG
    kinematic_chain = Chain.from_urdf_file(IKPY_URDF_PATH, base_elements=["panda_link0"]) 

    physics, passive_view, arm, gripper = construct_physics(cfg)

    # define controller
    osc = OSC(physics, arm, MOTOR_CONFIG["controllers"]["osc"])

    # define GA for tuning params
    
    def fitness_fn(params, controller):
        """
        Evaluate time to reach target and convergence.

        For now the target is fixed but in future test more diverse targets.
        """
        # overwrite control params
        controller.controller_gains["position"]["kp"] = float(params.at[0].get())
        controller.controller_gains["orientation"]["kp"] = float(params.at[2].get())
        #controller.controller_gains["position"]["damping_ratio"] = float(params.at[1].get())
        #controller.controller_gains["orientation"]["damping_ratio"] = float(params.at[3].get())

        # reset the simulation
        controller.physics.reset()
        controller.physics.data.qpos[:7] = np.array(cfg.robots.arm_configs.ready)

        # compute the eef targets
        target_eef_pose = np.array([0.45,0.0,0.6])
        target_orientation = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
        target_quat = tr.mat_to_quat(target_orientation)

        controller.eef_target_position = target_eef_pose
        controller.eef_target_velocity = np.zeros(3)
        controller.eef_target_quat = target_quat
        controller.eef_target_angular_velocity = np.zeros(3)
        
        # run controller to target
        start_time = controller.physics.data.time
        status = osc.run_controller(3.0)
        end_time = controller.physics.data.time
        
        # compute fitness
        time_to_target = end_time - start_time
        return time_to_target + (status*10000)

    rng = jax.random.PRNGKey(0)
    strategy = CMA_ES(
            popsize=10,
            num_dims=2,
            sigma_init=250,
            )
    params = strategy.params_strategy
    params = params.replace(init_min=150, init_max=250)
    state = strategy.initialize(rng, params)

    for t in range(10):
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_gen, state, params)
        print("x: ", x)
        print("state: ", state)
        fitness = []
        for param in x:
            fitness.append(fitness_fn(param, osc))
        state = strategy.tell(x, fitness, state, params)
    
    # get best params and fitness
    best_params = state.best_params
    best_fitness = state.best_fitness

    print("Best params: ", best_params)
    print("Best fitness: ", best_fitness)

