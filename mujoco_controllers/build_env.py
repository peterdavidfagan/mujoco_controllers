from typing import Tuple

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

from rearrangement_benchmark.environment.props import Rectangle
from hydra import compose, initialize
from hydra.utils import instantiate

def build_arena(name: str) -> composer.Arena:
    """Build a MuJoCo arena."""
    arena = empty.Arena(name=name)
    arena.mjcf_model.option.timestep = 0.001
    arena.mjcf_model.option.gravity = (0.0, 0.0, -9.8)
    arena.mjcf_model.option.noslip_iterations = 3
    arena.mjcf_model.size.nconmax = 1000
    arena.mjcf_model.size.njmax = 2000
    arena.mjcf_model.visual.__getattr__("global").offheight = 640
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    arena.mjcf_model.visual.map.znear = 0.0005
    return arena


def add_robot_and_gripper(arena: composer.Arena, arm, gripper) -> Tuple[composer.Entity, composer.Entity]:
    """Add a robot and gripper to the arena."""
    # attach the gripper to the robot
    robot.standard_compose(arm=arm, gripper=gripper)

    # define robot base site
    robot_base_site = arena.mjcf_model.worldbody.add(
        "site",
        name="robot_base",
        pos=(0.0, 0.0, 0.0),
    )

    # add the robot and gripper to the arena
    arena.attach(arm, robot_base_site)

    return arm, gripper

def construct_physics(cfg):
    # build the base arena
    arena = build_arena("base_scene")

    # add robot arm and gripper to the arena
    arm = instantiate(cfg.robots.arm)
    gripper = instantiate(cfg.robots.gripper)
    arm, gripper = add_robot_and_gripper(arena, arm, gripper)

    # add a block 
    rectangle = Rectangle(name="cube",
                          x_len=0.02,
                          y_len=0.02,
                          z_len=0.02,
                          rgba=(1.0, 0.0, 0.0, 1.0),
                          mass=0.1)
    frame = arena.add_free_entity(rectangle)
    rectangle.set_freejoint(frame.freejoint)


    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    # set the default arm joint positions to ready
    physics.data.qpos[:7] = np.array(cfg.robots.arm_configs.ready)
    rectangle.set_pose(physics, position=np.array([0.45,0.0,0.02]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]))

    # launch passive viewer
    passive_view = viewer.launch_passive(physics.model._model, physics.data._data)

    return physics, passive_view, arm, gripper
