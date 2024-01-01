from typing import Dict, Sequence, Tuple, List

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

import mujoco
from mujoco import viewer

from dm_control import composer, mjcf

from mujoco_controllers.models.arenas import empty
from mujoco_controllers.models.robot_arm import standard_compose

from hydra import compose, initialize
from hydra.utils import instantiate


# this prop class is take from dm_robotics: https://github.com/google-deepmind/dm_robotics/blob/main/py/moma/prop.py
# the rest of this file is custom code
class Prop(composer.Entity):
  """Base class for MOMA props."""

  def _build(self,
             name: str,
             mjcf_root: mjcf.RootElement,
             prop_root: str = 'prop_root'):
    """Base constructor for props.

    This constructor sets up the common observables and element access
    properties.

    Args:
      name: The unique name of this prop.
      mjcf_root: (mjcf.Element) The root element of the MJCF model.
      prop_root: (string) Name of the prop root body MJCF element.

    Raises:
      ValueError: If the model does not contain the necessary elements.
    """

    self._name = name
    self._mjcf_root = mjcf_root
    self._prop_root = mjcf_root.find('body', prop_root)  # type: mjcf.Element
    if self._prop_root is None:
      raise ValueError(f'model does not contain prop root {prop_root}.')
    self._freejoint = None  # type: mjcf.Element

  @property
  def name(self) -> str:
    return self._name

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to this prop."""
    return self._mjcf_root

  def set_pose(self, physics: mjcf.Physics, position: np.ndarray,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
               quaternion: np.ndarray) -> None:
    """Sets the pose of the prop wrt to where it was defined.

    This function overrides `Entity.set_pose`, which has the annoying property
    that it doesn't consider where the prop was originally attached.  EG if you
    do `pinch_site.attach(prop)`, the prop will be a sibling of pinch_site with
    the pinch_site's pose parameters, and calling
      `set_pose([0, 0, 0], [1, 0, 0, 0])`
    will clobber these params and move the prop to the parent-body origin.

    Oleg's fix uses an extra `prop_root` body that's a child of the sibling
    body, and sets the pose of this guy instead.

    Args:
      physics: An instance of `mjcf.Physics`.
      position: A NumPy array of size 3.
      quaternion: A NumPy array of size [w, i, j, k].

    Raises:
      RuntimeError: If the entity is not attached.
      Exception: If oleg isn't happy
    """

    if self._prop_root is None:
      raise Exception('prop {} missing root element'.format(
          self.mjcf_model.model))

    if self._freejoint is None:
      physics.bind(self._prop_root).pos = position  # pytype: disable=not-writable
      physics.bind(self._prop_root).quat = quaternion  # pytype: disable=not-writable
    else:
      # If we're attached via a freejoint then bind().pos or quat does nothing,
      # as the pose is controlled by qpos directly.
      physics.bind(self._freejoint).qpos = np.hstack([position, quaternion])  # pytype: disable=not-writable

  def set_freejoint(self, joint: mjcf.Element):
    """Associates a freejoint with this prop if attached to arena."""
    joint_type = joint.tag  # pytype: disable=attribute-error
    if joint_type != 'freejoint':
      raise ValueError(f'Expected a freejoint but received {joint_type}')
    self._freejoint = joint

  def disable_collisions(self) -> None:
    for geom in self.mjcf_model.find_all('geom'):
      geom.contype = 0
      geom.conaffinity = 0


class Rectangle(Prop):
    """Prop with a rectangular shape."""

    @staticmethod
    def _make(
        name:str,
        pos: Tuple[float, float, float]=(0.0, 0.0, 0.0),
        x_len: float = 0.1,
        y_len: float = 0.1,
        z_len: float = 0.1,
        rgba: Tuple[float, float, float,float]=(1, 0, 0, 1),
        solimp: Tuple[float, float, float]=(0.95, 0.995, 0.001),
        solref: Tuple[float, float, float]=(0.002, 0.7),
        mass: float = 0.1,
        margin: float = 0.15,
        gap: float = 0.15,
    ):
        """Make a block model: the mjcf element, and site."""
        mjcf_root = mjcf.element.RootElement(model=name)
        prop_root = mjcf_root.worldbody.add("body", name="prop_root")
        # add margin so object is pickable
        box = prop_root.add(
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=(x_len, y_len, z_len),
            solref=solref,
            solimp=solimp,
            condim=3,
            rgba=rgba,
            mass = mass,
            friction = "10 10 10",
            margin = margin,
            gap = gap,
        )
        site = prop_root.add(
            "site",
            name="box_centre",
            type="sphere",
            rgba=(0.1, 0.1, 0.1, 0.8),
            size=(0.005,),
            pos=(0, 0, 0),
            euler=(0, 0, 0),
        )  # Was (np.pi, 0, np.pi / 2)
        del box

        return mjcf_root, site

    def _build(  # pylint:disable=arguments-renamed
        self,
        rgba: List,
        name: str = "box",
        x_len: float = 0.1,
        y_len: float = 0.1,
        z_len: float = 0.1,
        pos=(0.0, 0.0, 0.0),
        solimp: Tuple[float, float, float]=(0.95, 0.995, 0.001),
        solref: Tuple[float, float, float]=(0.002, 0.7),
        mass: float = 0.1,
        margin: float = 0.15,
        gap: float = 0.15,
    ) -> None:
        mjcf_root, site = Rectangle._make(name,
                                          x_len=x_len,
                                          y_len=y_len,
                                          z_len=z_len,
                                          rgba=rgba,
                                          pos=pos,
                                          solimp=solimp,
                                          solref=solref,
                                          mass=mass,
                                          margin=margin,
                                          gap=gap,
                                          )
        super()._build(name, mjcf_root, "prop_root")
        del site

    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "red_rectangle_1",
        color: str = "red",
        min_object_size: float = 0.02,
        max_object_size: float = 0.05,     
        x_len: float = 0.04,
        y_len: float = 0.04,
        z_len: float = 0.04,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample_size: bool = False,
        sample_colour: bool = False,
        is_cube: bool = False,
        color_noise: float = 0.1,
    ) -> composer.Entity:
        """Add a block to the arena."""
        if sample_size:
            # sample block dimensions
            if is_cube:
                size = 3*[np.random.uniform(min_object_size, max_object_size)]
            else:
                size = np.random.uniform(min_object_size, max_object_size, size=3)

            x_len, y_len, z_len = size[0], size[1], size[2]
                
        if sample_colour:
            # sample block color
            rgba = COLOURS[color]
            # add noise
            rgba = [ c + np.random.uniform(-color_noise, color_noise) for c in rgba]
            rgba[3] = 1.0
            
        # create block and add to arena
        rectangle = Rectangle(name=name,
                              x_len=x_len,
                              y_len=y_len,
                              z_len=z_len,
                              rgba=rgba)
        frame = arena.add_free_entity(rectangle)
        rectangle.set_freejoint(frame.freejoint)

        return rectangle

def build_arena(name: str) -> composer.Arena:
    """Build a MuJoCo arena."""
    arena = empty.Arena(name=name)
    arena.mjcf_model.option.timestep = 0.0005
    arena.mjcf_model.option.gravity = (0.0, 0.0, -9.8)
    arena.mjcf_model.option.noslip_iterations = 3
    arena.mjcf_model.size.nconmax = 1000
    arena.mjcf_model.size.njmax = 2000
    arena.mjcf_model.visual.__getattr__("global").offheight = 640
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    arena.mjcf_model.visual.map.znear = 0.0005
    return arena


def construct_physics(cfg):
    # build the base arena
    arena = build_arena("base_scene")

    # add robot arm and gripper to the arena
    arm = instantiate(cfg.robots.arm.arm)
    gripper = instantiate(cfg.robots.end_effector.end_effector)
    standard_compose(arm=arm, gripper=gripper)
    robot_base_site = arena.mjcf_model.worldbody.add(
        "site",
        name="robot_base",
        pos=(0.0, 0.0, 0.0),
    )
    arena.attach(arm, robot_base_site)

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
    physics.data.qpos[:7] = np.array(cfg.robots.arm.default_configurations.home)
    rectangle.set_pose(physics, position=np.array([0.45,0.0,0.02]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]))

    # launch passive viewer
    passive_view = viewer.launch_passive(physics.model._model, physics.data._data)

    return physics, passive_view, arm, gripper
