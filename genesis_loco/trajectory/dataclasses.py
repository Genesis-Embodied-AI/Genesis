from __future__ import annotations

import os.path
from dataclasses import dataclass, fields, asdict, replace
import pickle

import flax.serialization
import numpy as np
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import mujoco
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
from typing import Union
from types import ModuleType

from loco_mujoco.core.observations.base import ObservationContainer


@dataclass
class Trajectory:
    """
    Main data structure to store the trajectories.

    Args:
        info (TrajectoryInfo): Static information about the trajectory. This includes the joint names, frequency,
            body names, site names as well a reduced version of the Mujoco model.
        data (TrajectoryData): Dynamic information about the trajectory. This includes the qpos, qvel, xpos, xquat etc.
        transitions (TrajectoryTransitions): Trajectory transitions used for training RL algorithms where the trajectory
            consists of tuples of (observation, action, reward, next_observation, absorbing, done). (optional)
        obs_container (ObservationContainer): The observation container contains all information needed to build an
            observation from Mujoco data/model. (optional)
    """

    info: TrajectoryInfo
    data: TrajectoryData
    transitions: TrajectoryTransitions = None
    obs_container: ObservationContainer = None

    @staticmethod
    def concatenate(trajs: list, backend: ModuleType = jnp):
        traj_data = [traj.data for traj in trajs]
        traj_info = [traj.info for traj in trajs]
        traj_data, traj_info = TrajectoryData.concatenate(traj_data, traj_info, backend)
        return Trajectory(info=traj_info, data=traj_data)

    def to_dict(self):
        """
        Serializes the trajectory to dict.

        Returns:
            A dictionary containing the trajectory data.

        """
        serialized = flax.serialization.to_state_dict(self.data)
        traj_info_dict = self.info.to_dict()
        traj_model = flax.serialization.to_state_dict(traj_info_dict["model"])
        del traj_info_dict["model"]
        traj_transitions = flax.serialization.to_state_dict(self.transitions)
        serialized |= traj_info_dict
        serialized |= traj_model
        if self.transitions is not None:
            serialized |= traj_transitions
        if self.obs_container is not None:
            obs_container = pickle.dumps(self.obs_container)
            serialized["obs_container"] = obs_container
        return serialized

    def save(self, path: str) -> None:
        """
        Serializes the trajectory and saves it to a npz file.

        Args:
            path (str): Path to save the trajectory.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        serialized = self.to_dict()
        np.savez(str(path), **serialized)

    @classmethod
    def load(cls, path, backend: ModuleType = jnp):
        """
        Loads a trajectory from a npz file.

        Args:
            path (str): Path to the trajectory to load.
            backend: Backend to use for arrays. Either numpy or jax.numpy.

        Returns:
            A new instance of Trajectory.
        """

        def is_none_object_array(array):
            # Check if the input is an instance of np.ndarray
            if isinstance(array, np.ndarray):
                # Check if the dtype is object and all elements are None
                return array.dtype == object and np.all(array == None)
            return False

        data = np.load(path, allow_pickle=True)
        converted_info = {}
        converted_model = {}
        converted_data = {}
        converted_transitions = {}
        converted_obs_container = None
        for key, value in data.items():
            if key in TrajectoryInfo.get_attribute_names():
                converted_info[key] = None if is_none_object_array(value) else value.tolist()
            elif key in TrajectoryModel.get_attribute_names():
                converted_model[key] = None if is_none_object_array(value) else backend.array(value)
            elif key in TrajectoryData.get_attribute_names():
                converted_data[key] = backend.array(value)
            elif key in TrajectoryTransitions.get_attribute_names():
                converted_transitions[key] = backend.array(value)
            elif key == "obs_container":
                converted_obs_container = value
            else:
                raise ValueError(f"Unknown key {key} in the npz file.")

        _all = {"data": TrajectoryData(**converted_data),
                "info": TrajectoryInfo(model=TrajectoryModel(**converted_model), **converted_info)}
        if converted_transitions:
            _all["transitions"] = TrajectoryTransitions(**converted_transitions)
        if converted_obs_container:
            _all["obs_container"] = pickle.loads(converted_obs_container)
        return cls(**_all)


@dataclass
class TrajectoryInfo:
    """
    Data structure to store the trajectory information.
    """
    joint_names: list[str]
    model: TrajectoryModel
    frequency: float
    body_names: list[str] = None
    site_names: list[str] = None
    metadata: dict = None

    def __post_init__(self):
        self.joint_name2ind_qpos = {}
        self.joint_name2ind_qvel = {}
        j_qpos = 0
        j_qvel = 0
        for i, item in enumerate(zip(self.joint_names, self.model.jnt_type)):
            j_name, j_type = item
            if j_type == mujoco.mjtJoint.mjJNT_FREE:
                self.joint_name2ind_qpos[j_name] = np.arange(j_qpos, j_qpos + 7)
                self.joint_name2ind_qvel[j_name] = np.arange(j_qvel, j_qvel + 6)
                j_qpos += 7
                j_qvel += 6
            elif j_type == mujoco.mjtJoint.mjJNT_SLIDE or j_type == mujoco.mjtJoint.mjJNT_HINGE:
                self.joint_name2ind_qpos[j_name] = np.array([j_qpos])
                self.joint_name2ind_qvel[j_name] = np.array([j_qvel])
                j_qpos += 1
                j_qvel += 1
            else:
                raise ValueError(f"Unsupported joint type: {j_type} for joint {j_name}")

        self.body_name2ind = {}
        if self.body_names is not None:
            for i, b_name in enumerate(self.body_names):
                self.body_name2ind[b_name] = np.array([i])

        self.site_name2ind = {}
        if self.site_names is not None:
            for i, s_name in enumerate(self.site_names):
                self.site_name2ind[s_name] = np.array([i])

    def __eq__(self, other, backend: ModuleType = jnp):
        if not isinstance(other, TrajectoryInfo):
            return False

        # Compare joint_name2ind_qpos dictionaries
        if self.joint_name2ind_qpos.keys() != other.joint_name2ind_qpos.keys():
            return False
        for key in self.joint_name2ind_qpos:
            if not backend.array_equal(self.joint_name2ind_qpos[key], other.joint_name2ind_qpos[key]):
                return False

        # Compare joint_name2ind_qvel dictionaries
        if self.joint_name2ind_qvel.keys() != other.joint_name2ind_qvel.keys():
            return False
        for key in self.joint_name2ind_qvel:
            if not backend.array_equal(self.joint_name2ind_qvel[key], other.joint_name2ind_qvel[key]):
                return False

        # Compare body_name2ind dictionaries
        if self.body_name2ind.keys() != other.body_name2ind.keys():
            return False
        for key in self.body_name2ind:
            if not backend.array_equal(self.body_name2ind[key], other.body_name2ind[key]):
                return False

        # Compare site_name2ind dictionaries
        if self.site_name2ind.keys() != other.site_name2ind.keys():
            return False
        for key in self.site_name2ind:
            if not backend.array_equal(self.site_name2ind[key], other.site_name2ind[key]):
                return False

        # Compare other attributes
        return (
                self.joint_names == other.joint_names
                and self.frequency == other.frequency
                and self.model == other.model
                and self.body_names == other.body_names
                and self.site_names == other.site_names
                and self.metadata == other.metadata
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def get_attribute_names(cls):
        return [field.name for field in fields(cls)]

    def add_joint(self, joint_name, joint_type, backend: ModuleType = jnp):
        """
        Add a new joint to the trajectory info.

        Args:
            joint_name (list(str)): Joint name to add.
            joint_type (mujoco.mjtJoint): Joint type to add.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
            A new instance of TrajectoryInfo with the specified joint added.
        """
        assert isinstance(joint_name, str)
        joint_type = int(joint_type)

        new_model = self.model.add_joint(joint_type, backend)
        return replace(self,
                       joint_names=self.joint_names + [joint_name],
                       model=new_model
                       )

    def add_body(self, body_name, body_rootid, body_weldid, body_mocapid,
                 body_pos, body_quat, body_ipos, body_iquat, backend: ModuleType = jnp):
        """
        Add a new body to the trajectory info.

        Args:
            body_name (list(str)): Body name to add.
            body_rootid (int): Root id of the new body.
            body_weldid (int): Weld id of the new body.
            body_mocapid (int): Mocap id of the new body.
            body_pos (Array): Position of the new body.
            body_quat (Array): Quaternion of the new body.
            body_ipos (Array): Initial position of the new body.
            body_iquat (Array): Initial quaternion of the new body.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
            A new instance of TrajectoryInfo with the specified body added.
        """

        new_model = self.model.add_body(body_rootid, body_weldid, body_mocapid, body_pos,
                                        body_quat, body_ipos, body_iquat, backend)

        return replace(self,
                       body_names=self.body_names + [body_name],
                       model=new_model
                       )

    def add_site(self, site_name, site_pos, site_quat, site_bodyid, backend: ModuleType = jnp):
        """
        Add a new site to the trajectory info.

        Args:
            site_name (list[str]): site name to add.
            site_pos (Array): Position of the new site.
            site_quat (Array): Quaternion of the new site.
            site_bodyid (int): Body id of the new site.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
            A new instance of TrajectoryInfo with the specified site added.
        """
        assert isinstance(site_name, str)
        assert isinstance(site_pos, jax.Array) or isinstance(site_pos, np.ndarray)
        assert isinstance(site_quat, jax.Array) or isinstance(site_quat, np.ndarray)

        new_model = self.model.add_site(site_pos, site_quat, site_bodyid, backend)

        return replace(self,
                       site_names=self.site_names + [site_name],
                       model=new_model
                       )

    def remove_joints(self, joint_names, backend: ModuleType = jnp):
        """
        Remove the joints with the specified ids from the trajectory info.

        Args:
            joint_names (list[str]): List of joint names to remove.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
            A new instance of TrajectoryInfo with the specified joints removed.
        """
        new_model = self.model.remove_joints(backend.array([self.joint_names.index(name) for name in joint_names])
                                             , backend)
        return replace(self,
                       joint_names=[name for name in self.joint_names if name not in joint_names],
                       model=new_model
                       )

    def remove_bodies(self, body_names, backend: ModuleType = jnp):
        """
        Remove the bodies with the specified ids from the trajectory info.

        Args:
            body_names (list[str]): List of body ids to remove.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
            A new instance of TrajectoryInfo with the specified bodies removed.
        """
        new_model = self.model.remove_bodies(backend.array([self.body_name2ind[name] for name in body_names]),
                                             backend)
        return replace(self,
                       body_names=[name for name in self.body_names if name not in body_names],
                       model=new_model
                       )

    def remove_sites(self, site_names, backend: ModuleType = jnp):
        """
        Remove the sites with the specified ids from the trajectory info.

        Args:
            site_names (list[str]): List of site ids to remove.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
            A new instance of TrajectoryInfo with the specified sites removed.
        """
        new_model = self.model.remove_sites(backend.array([self.site_name2ind[name] for name in site_names]), backend)
        return replace(self,
                       site_names=[name for name in self.site_names if name not in site_names],
                       model=new_model
                       )

    def reorder_joints(self, new_order, backend: ModuleType = jnp):
        """

        Args:
            new_order (list[int]): List of indices of new joint order.
            backend (Union[jax, numpy]): Backend to use for the computation.
        """
        new_model = self.model.reorder_joints(new_order, backend)
        return replace(self,
                       joint_names=[self.joint_names[i] for i in new_order],
                       model=new_model
                       )

    def reorder_bodies(self, new_order, backend: ModuleType = jnp):
        """

        Args:
            new_order (list[int]): List of indices of new body order.
            backend (Union[jax, numpy]): Backend to use for the computation.
        """
        new_model = self.model.reorder_bodies(new_order, backend)
        return replace(self,
                       body_names=[self.body_names[i] for i in new_order],
                       model=new_model
                       )

    def reorder_sites(self, new_order, backend: ModuleType = jnp):
        """

        Args:
            new_order (list[int]): List of indices of new site order.
            backend (Union[jax, numpy]): Backend to use for the computation.
        """
        new_model = self.model.reorder_sites(new_order, backend)
        return replace(self,
                       site_names=[self.site_names[i] for i in new_order],
                       model=new_model
                       )


@struct.dataclass
class TrajectoryModel:
    """
    Data structure to store relevant attributes of the Mujoco model.
    """

    # joint properties in Mujoco model
    njnt: int
    jnt_type: Union[jax.Array, np.ndarray]

    # body properties in Mujoco model
    nbody: int = None
    body_rootid: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    body_weldid: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    body_mocapid: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    body_pos: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    body_quat: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    body_ipos: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    body_iquat: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))

    # site properties in Mujoco model
    nsite: int = None
    site_bodyid: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    site_pos: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    site_quat: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))

    def __eq__(self, other, backend: ModuleType = jnp):
        if not isinstance(other, TrajectoryModel):
            return False

        return (
            self.njnt == other.njnt
            and backend.array_equal(self.jnt_type, other.jnt_type)
            and self.nbody == other.nbody
            and backend.array_equal(self.body_rootid, other.body_rootid)
            and backend.array_equal(self.body_weldid, other.body_weldid)
            and backend.array_equal(self.body_mocapid, other.body_mocapid)
            and backend.array_equal(self.body_pos, other.body_pos)
            and backend.array_equal(self.body_quat, other.body_quat)
            and backend.array_equal(self.body_ipos, other.body_ipos)
            and backend.array_equal(self.body_iquat, other.body_iquat)
            and self.nsite == other.nsite
            and backend.array_equal(self.site_bodyid, other.site_bodyid)
            and backend.array_equal(self.site_pos, other.site_pos)
            and backend.array_equal(self.site_quat, other.site_quat)
        )

    def add_joint(self, jnt_type, backend: ModuleType = jnp):
        """
        Add a new joint to the trajectory model.

        Args:
            jnt_type: Type of the new joint.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
        A new instance of TrajectoryModel with the new joint added.
        """
        return self.replace(
            njnt=self.njnt + 1,
            jnt_type=backend.concatenate([self.jnt_type, backend.array([jnt_type])])
        )

    def add_body(self, body_rootid, body_weldid, body_mocapid, body_pos, body_quat,
                 body_ipos, body_iquat, backend: ModuleType = jnp):
        """
        Add a new body to the trajectory model.

        Args:
            body_rootid (Array): Root id of the new body.
            body_weldid (Array): Weld id of the new body.
            body_mocapid (Array): Mocap id of the new body.
            body_pos (Array): Position of the new body.
            body_quat (Array): Quaternion of the new body.
            body_ipos (Array): Initial position of the new body.
            body_iquat (Array): Initial quaternion of the new body.
            backend (Union[jax, numpy]): Backend to use for the computation.

        Returns:
        A new instance of TrajectoryModel with the new body added.
        """
        return self.replace(
            nbody=self.nbody + 1,
            body_rootid=backend.concatenate([self.body_rootid, backend.array([body_rootid])]),
            body_weldid=backend.concatenate([self.body_weldid, backend.array([body_weldid])]),
            body_mocapid=backend.concatenate([self.body_mocapid, backend.array([body_mocapid])]),
            body_pos=backend.concatenate([self.body_pos, backend.array([body_pos])]),
            body_quat=backend.concatenate([self.body_quat, backend.array([body_quat])]),
            body_ipos=backend.concatenate([self.body_ipos, backend.array([body_ipos])]),
            body_iquat=backend.concatenate([self.body_iquat, backend.array([body_iquat])])
        )

    def add_site(self, site_pos, site_quat, site_body_id, backend: ModuleType = jnp):
        """
        Add a new site to the trajectory model.

        Args:
            site_pos (Array): Position of the new site.
            site_quat (Array): Quaternion of the new site.
            backend (ModuleType): Backend to use for the computation.

        Returns:
        A new instance of TrajectoryModel with the new site added.
        """
        return self.replace(
            nsite=self.nsite + 1,
            site_bodyid=backend.concatenate([self.site_bodyid, backend.array([site_body_id])]),
            site_pos=backend.concatenate([self.site_pos, backend.array([site_pos])]),
            site_quat=backend.concatenate([self.site_quat, backend.array([site_quat])],)
        )

    def remove_joints(self, joint_ids, backend: ModuleType = jnp):
        """
        Remove the joints with the specified ids from the trajectory model.

        Args:
            joint_ids (Union[jax.Array, np.ndarray]): Array of joint ids to remove.
            backend (ModuleType): Backend to use for the computation.

        Returns:
        A new instance of TrajectoryModel with the specified joints removed.
        """
        return self.replace(
            njnt=self.njnt - len(joint_ids),
            jnt_type=backend.delete(self.jnt_type, joint_ids, axis=0)
        )

    def remove_bodies(self, body_ids, backend: ModuleType = jnp):
        """
        Remove the bodies with the specified ids from the trajectory model.

        Args:
            body_ids (jax.Array): Array of body ids to remove.
            backend (ModuleType): Backend to use for the computation.

        Returns:
        A new instance of TrajectoryModel with the specified bodies removed.
        """
        return self.replace(
            nbody=self.nbody - len(body_ids),
            body_rootid=backend.delete(self.body_rootid, body_ids, axis=0),
            body_weldid=backend.delete(self.body_weldid, body_ids, axis=0),
            body_mocapid=backend.delete(self.body_mocapid, body_ids, axis=0),
            body_pos=backend.delete(self.body_pos, body_ids, axis=0),
            body_quat=backend.delete(self.body_quat, body_ids, axis=0),
            body_ipos=backend.delete(self.body_ipos, body_ids, axis=0),
            body_iquat=backend.delete(self.body_iquat, body_ids, axis=0)
        )

    def remove_sites(self, site_ids, backend: ModuleType = jnp):
        """
        Remove the sites with the specified ids from the trajectory model.

        Args:
            site_ids (Union[jax.Array, np.ndarray]): Array of site ids to remove.
            backend (ModuleType): Backend to use for the computation.

        Returns:
        A new instance of TrajectoryModel with the specified sites removed.
        """
        return self.replace(
            nsite=self.nsite - len(site_ids),
            site_bodyid=backend.delete(self.site_bodyid, site_ids, axis=0),
            site_pos=backend.delete(self.site_pos, site_ids, axis=0),
            site_quat=backend.delete(self.site_quat, site_ids, axis=0)
        )

    def reorder_joints(self, new_order, backend: ModuleType = jnp):
        """

        Args:
            new_order (list[int]): List of indices of new joint order.
            backend (ModuleType): Backend to use for the computation.

        """
        new_order = backend.array(new_order)
        return self.replace(
            jnt_type=self.jnt_type[new_order]
        )

    def reorder_bodies(self, new_order, backend: ModuleType = jnp):
        """

        Args:
            new_order (list[int]): List of indices of new body order.
            backend (ModuleType): Backend to use for the computation.
        """
        new_order = backend.array(new_order)
        return self.replace(
            body_rootid=self.body_rootid[new_order],
            body_weldid=self.body_weldid[new_order],
            body_mocapid=self.body_mocapid[new_order],
            body_pos=self.body_pos[new_order],
            body_quat=self.body_quat[new_order],
            body_ipos=self.body_ipos[new_order],
            body_iquat=self.body_iquat[new_order]
        )

    def reorder_sites(self, new_order, backend: ModuleType = jnp):
        """

        Args:
            new_order (list[int]): List of indices of new site order.
            backend (ModuleType): Backend to use for the computation.
        """
        new_order = backend.array(new_order)
        return self.replace(
            site_bodyid=self.site_bodyid[new_order],
            site_pos=self.site_pos[new_order],
            site_quat=self.site_quat[new_order]
        )

    @classmethod
    def get_attribute_names(cls):
        return [field.name for field in fields(cls)]

    def to_numpy(self):
        dic = flax.serialization.to_state_dict(self)
        for key, value in dic.items():
            dic[key] = np.array(value) if (isinstance(value, jax.Array) or isinstance(value, np.ndarray)) else value
        return TrajectoryModel(**dic)

    def to_jax(self):
        dic = flax.serialization.to_state_dict(self)
        for key, value in dic.items():
            dic[key] = jnp.array(value) if (isinstance(value, jax.Array) or isinstance(value, np.ndarray)) else value
        return TrajectoryModel(**dic)


@struct.dataclass
class SingleData:
    """
    Data structure to store relevant attributes of Mujoco Data. This data structure is supposed to be a reduced version
    of the Mujoco data structure to reduce memory.

    While it currently stores just a few elements, it can be extended to store more elements in the future.
    """
    # joint properties in Mujoco datastructure
    qpos: Union[jax.Array, np.ndarray]
    qvel: Union[jax.Array, np.ndarray]

    # global body properties
    xpos: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    xquat: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    cvel: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    subtree_com: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))

    # global site properties
    site_xpos: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    site_xmat: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))

    @property
    def is_complete(self):
        return all(getattr(self, field).size > 0 for field in self.__dataclass_fields__)


@struct.dataclass
class TrajectoryData(SingleData):
    """
    Data structure to store the trajectory data. It holds everything in SingleData, but with an additional dimension
    (the batch dimension for an arbitrary amount of samples and trajectories).
     It also includes the split_points attribute to separate the different trajectories.

    Note 1: All samples are stacked along the first dimension, and the split_points attribute is used to separate
    the different trajectories. The split_points attribute is a list of indices that define the beginning of each
    trajectory, and the end of the last trajectory.

    Note 2: This datastructure is meant to be used with jax arrays. However, a conversion to numpy is implemented, but
    not recommended.

    """

    # points defining the beginning of each trajectory, and the end of the last trajectory
    split_points: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))

    def __eq__(self, other, backend: ModuleType = jnp):
        if not isinstance(other, TrajectoryData):
            return False

        # Compare attributes from SingleData
        if not (backend.array_equal(self.qpos, other.qpos) and
                backend.array_equal(self.qvel, other.qvel) and
                backend.array_equal(self.xpos, other.xpos) and
                backend.array_equal(self.xquat, other.xquat) and
                backend.array_equal(self.cvel, other.cvel) and
                backend.array_equal(self.subtree_com, other.subtree_com) and
                backend.array_equal(self.site_xpos, other.site_xpos) and
                backend.array_equal(self.site_xmat, other.site_xmat) and
                backend.array_equal(self.split_points, other.split_points)):
            return False
        return True

    def get(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        """
        Retrieve the corresponding data for a given trajectory index and sub-trajectory index.

        Args:
            traj_index: Index of the trajectory.
            sub_traj_index: Index of the sub-trajectory.
            backend: Backend to use for the computation.

        Returns:
        A new instance of SingleData containing the indexed data.
        """

        # Get the start indices for the selected trajectory
        start_idx = self.split_points[traj_index]

        # Get the beginning and end of the slice
        ind = start_idx + sub_traj_index

        return SingleData(
            qpos=backend.squeeze(self.qpos[ind].copy()),
            qvel=backend.squeeze(self.qvel[ind].copy()),
            xpos=backend.squeeze(self.xpos[ind].copy()) if self.xpos.size > 0 else backend.empty((1, 0)),
            xquat=backend.squeeze(self.xquat[ind].copy()) if self.xquat.size > 0 else backend.empty((1, 0)),
            cvel=backend.squeeze(self.cvel[ind].copy()) if self.cvel.size > 0 else backend.empty((1, 0)),
            subtree_com=backend.squeeze(self.subtree_com[ind].copy()) if self.subtree_com.size > 0 else backend.empty((1, 0)),
            site_xpos=backend.squeeze(self.site_xpos[ind].copy()) if self.site_xpos.size > 0 else backend.empty((1, 0)),
            site_xmat=backend.squeeze(self.site_xmat[ind].copy()) if self.site_xmat.size > 0 else backend.empty((1, 0))
        )

    @classmethod
    def dynamic_slice_in_dim(cls, data, traj_index, sub_traj_start_index, slice_length, backend: ModuleType = jnp):

        # Get the start indices for the selected trajectory
        start_idx = data.split_points[traj_index]

        # Get the beginning and end of the slice
        slice_start = start_idx + sub_traj_start_index

        # Apply dynamic slicing to the trajectory data
        return data.replace(
            qpos=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.qpos, slice_start, slice_length, backend)),
            qvel=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.qvel, slice_start, slice_length, backend)),
            xpos=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.xpos, slice_start, slice_length, backend)) if data.xpos.size > 0 else backend.empty(0),
            xquat=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.xquat, slice_start, slice_length, backend)) if data.xquat.size > 0 else backend.empty(0),
            cvel=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.cvel, slice_start, slice_length, backend)) if data.cvel.size > 0 else backend.empty(0),
            subtree_com=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.subtree_com, slice_start, slice_length, backend)) if data.subtree_com.size > 0 else backend.empty(0),
            site_xpos=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.site_xpos, slice_start, slice_length, backend)) if data.site_xpos.size > 0 else backend.empty(0),
            site_xmat=backend.squeeze(cls._dynamic_slice_in_dim_compat(data.site_xmat, slice_start, slice_length, backend)) if data.site_xmat.size > 0 else backend.empty(0),
            split_points=backend.array([0, slice_length])
        )

    @classmethod
    def _dynamic_slice_in_dim_compat(cls, arr, start, length, backend):
        if backend == jnp:
            return cls._jax_dynamic_slice_in_dim(arr, start, length)
        else:
            return cls._np_dynamic_slice_in_dim(arr, start, length)

    @staticmethod
    def _jax_dynamic_slice_in_dim(arr, start, length):
        return lax.dynamic_slice_in_dim(arr, start, length)

    @staticmethod
    def _np_dynamic_slice_in_dim(arr, start, length):
        return arr[start:start+length].copy()

    @staticmethod
    def _get_single_attribute(attribute, split_points, traj_index, sub_traj_index, backend):
        """
        Helper function to extract a single attribute.
        """
        # Calculate start index
        start_idx = split_points[traj_index] + sub_traj_index
        return backend.squeeze(attribute[start_idx].copy())

    def _dynamic_slice_in_dim_single(self, attribute, split_points, traj_index, sub_traj_index, slice_length, backend):
        """
        Helper function to extract a single attribute slice.
        """
        # Calculate start index
        start_idx = split_points[traj_index]

        # Calculate the slice start index based on the sub-trajectory index
        slice_start = start_idx + sub_traj_index

        # Slice the desired attribute using `lax.dynamic_slice_in_dim`
        return backend.squeeze(self._dynamic_slice_in_dim_compat(attribute, slice_start, slice_length, backend))

    def get_qpos(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.qpos, self.split_points, traj_index, sub_traj_index, backend)

    def get_qvel(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.qvel, self.split_points, traj_index, sub_traj_index, backend)

    def get_xpos(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.xpos, self.split_points, traj_index, sub_traj_index, backend)

    def get_xquat(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.xquat, self.split_points, traj_index, sub_traj_index, backend)

    def get_cvel(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.cvel, self.split_points, traj_index, sub_traj_index, backend)

    def get_subtree_com(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.subtree_com, self.split_points, traj_index, sub_traj_index, backend)

    def get_site_xpos(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.site_xpos, self.split_points, traj_index, sub_traj_index, backend)

    def get_site_xmat(self, traj_index, sub_traj_index, backend: ModuleType = jnp):
        return self._get_single_attribute(self.site_xmat, self.split_points, traj_index, sub_traj_index, backend)

    def get_qpos_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.qpos, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def get_qvel_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.qvel, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def get_xpos_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.xpos, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def get_xquat_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.xquat, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def get_cvel_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.cvel, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def get_subtree_com_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.subtree_com, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def get_site_xpos_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.site_xpos, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def get_site_xmat_slice(self, traj_index, sub_traj_index, slice_length, backend: ModuleType = jnp):
        return self._dynamic_slice_in_dim_single(self.site_xmat, self.split_points,
                                                 traj_index, sub_traj_index, slice_length, backend)

    def add_joint(self, qpos_value=0.0, qvel_value=0.0, backend: ModuleType = jnp):
        """
        Adds a new joint with a default value to the trajectory data.

        Args:
            qpos_value (float): Default position value for the trajectory of the new joint.
            qvel_value (float): Default velocity value for the trajectory of the new joint.
            backend: Backend to use for the computation.

        Returns:
            A new instance of TrajectoryData with the new joint added.

        """
        return self.replace(
            qpos=backend.concatenate([self.qpos, backend.full((self.qpos.shape[0], 1), qpos_value)], axis=1),
            qvel=backend.concatenate([self.qvel, backend.full((self.qvel.shape[0], 1), qvel_value)], axis=1)
        )

    def add_body(self, xpos_value=0.0, cvel_value=0.0, subtree_com_value=0.0, backend: ModuleType = jnp):
        """
        Adds a new body with a default value to the trajectory data.

        Args:
            xpos_value (float): Default position value for the trajectory of the new body.
            cvel_value (float): Default velocity value for the trajectory of the new body.
            subtree_com_value (float): Default subtree com value for the trajectory of the new body.
            backend: Backend to use for the computation.

        Returns:
            A new instance of TrajectoryData with the new body added.
        """
        quats = backend.broadcast_to(backend.array([1.0, 0.0, 0.0, 0.0]), (self.xquat.shape[0], 1, 4))
        return self.replace(
            xpos=backend.concatenate([self.xpos, backend.full((self.xpos.shape[0], 1, 3), xpos_value)], axis=1),
            xquat=backend.concatenate([self.xquat, quats], axis=1),
            cvel=backend.concatenate([self.cvel, backend.full((self.cvel.shape[0], 1, 6), cvel_value)], axis=1),
            subtree_com=backend.concatenate([self.subtree_com, backend.full((self.subtree_com.shape[0], 1, 3),
                                                                            subtree_com_value)], axis=1)
        )

    def add_site(self, site_xpos_value=0.0, backend: ModuleType = jnp):
        """
        Adds a new site with a default value for the position/velocity and an identity matrix as a
        rotation to the trajectory data.

        Args:
            site_xpos_value (float): Default position value for the trajectory of the new site.
            backend: Backend to use for the computation.

        Returns:
            A new instance of TrajectoryData with the new site added.

        """
        return self.replace(
            site_xpos=backend.concatenate([self.site_xpos, backend.full((self.site_xpos.shape[0], 1, 3),
                                                                site_xpos_value)], axis=1),
            site_xmat=backend.concatenate([self.site_xmat, backend.broadcast_to(backend.eye(3).flatten(),
                                                                                (self.site_xmat.shape[0], 1, 9))], axis=1)
        )

    def remove_joints(self, joint_qpos_ids, joint_qvel_ids, backend: ModuleType = jnp):
        """
        Remove the joints with the specified ids from the trajectory data.

        Args:
            joint_qpos_ids (Union[jax.Array, np.ndarray]): Array of joint qpos ids to remove.
            joint_qvel_ids (Union[jax.Array, np.ndarray]): Array of joint qvel ids to remove.
            backend: Backend to use for the computation.

        Returns:
            A new instance of TrajectoryData with the specified joints removed.
        """

        # Remove the specified joints from the trajectory data
        return self.replace(
            qpos=backend.delete(self.qpos, joint_qpos_ids, axis=1),
            qvel=backend.delete(self.qvel, joint_qvel_ids, axis=1)
        )

    def remove_bodies(self, body_ids, backend: ModuleType = jnp):
        """
        Remove the bodies with the specified ids from the trajectory data.

        Args:
            body_ids (Union[jax.Array, np.ndarray]): Array of body ids to remove.
            backend: Backend to use for the computation.

        Returns:
            A new instance of TrajectoryData with the specified bodies removed.
        """

        # Remove the specified bodies from the trajectory data
        return self.replace(
            xpos=backend.delete(self.xpos, body_ids, axis=1),
            xquat=backend.delete(self.xquat, body_ids, axis=1),
            cvel=backend.delete(self.cvel, body_ids, axis=1),
            subtree_com=backend.delete(self.subtree_com, body_ids, axis=1)
        )

    def remove_sites(self, site_ids, backend: ModuleType = jnp):
        """
        Remove the sites with the specified ids from the trajectory data.

        Args:
            site_ids (Union[jax.Array, np.ndarray]): Array of site ids to remove.
            backend: Backend to use for the computation.

        Returns:
            A new instance of TrajectoryData with the specified sites removed.
        """
        # Remove the specified sites from the trajectory data
        return self.replace(
            site_xpos=backend.delete(self.site_xpos, site_ids, axis=1),
            site_xmat=backend.delete(self.site_xmat, site_ids, axis=1)
        )

    def reorder_joints(self, new_order_qpos, new_order_qvel):
        """
        Reorder the joints in the trajectory data.

        Args:
            new_order_qpos (Union[jax.Array, np.ndarray]): Array of indices specifying the new order of the joints positions.
            new_order_qvel (Union[jax.Array, np.ndarray]): Array of indices specifying the new order of the joints velocities.

        Returns:
            A new instance of TrajectoryData with the joints reordered.
        """
        return self.replace(
            qpos=self.qpos[:, new_order_qpos],
            qvel=self.qvel[:, new_order_qvel]
        )

    def reorder_bodies(self, new_order):
        """
        Reorder the bodies in the trajectory data.

        Args:
            new_order (Union[jax.Array, np.ndarray]): Array of indices specifying the new order of the bodies.

        Returns:
            A new instance of TrajectoryData with the bodies reordered.
        """
        return self.replace(
            xpos=self.xpos[:, new_order],
            xquat=self.xquat[:, new_order],
            cvel=self.cvel[:, new_order],
            subtree_com=self.subtree_com[:, new_order]
        )

    def reorder_sites(self, new_order):
        """
        Reorder the sites in the trajectory data.

        Args:
            new_order (Union[jax.Array, np.ndarray]): Array of indices specifying the new order of the sites.

        Returns:
            A new instance of TrajectoryData with the sites reordered.
        """
        return self.replace(
            site_xpos=self.site_xpos[:, new_order],
            site_xmat=self.site_xmat[:, new_order]
        )

    @staticmethod
    def concatenate(traj_datas: list, traj_infos: list, backend: ModuleType = jnp):
        """
        Concatenate a list of TrajectoryData instances given that the TrajectoryInfos are equivalent.

        Args:
            traj_datas (list): List of TrajectoryData instances to concatenate.
            traj_infos (list): List of TrajectoryInfo instances to concatenate.
            backend: Backend to use for the computation.

        Returns:
        New instance of TrajectoryData and TrajectoryInfo containing the concatenated data.
        """
        assert len(traj_datas) == len(traj_infos), "TrajectoryData and TrajectoryInfo must have the same length!"

        # assert that all traj_infos are equivalent
        assert all([info == traj_infos[0] for info in traj_infos]), ("TrajectoryInfos must be equivalent"
                                                                     " for concatenation!")

        # create new TrajectoryData
        new_split_points = []
        curr_n_samples = 0
        for i, data in enumerate(traj_datas):
            split_points = data.split_points
            if backend == jnp:
                split_points = split_points.at[:].add(curr_n_samples)
            else:
                split_points = split_points + curr_n_samples
            curr_n_samples = split_points[-1]
            new_split_points.append(split_points[:-1])

        new_split_points = backend.concatenate(new_split_points + [backend.array([curr_n_samples])], axis=0)

        # Concatenate the trajectory data
        new_traj_data = TrajectoryData(
            qpos=backend.concatenate([data.qpos for data in traj_datas], axis=0),
            qvel=backend.concatenate([data.qvel for data in traj_datas], axis=0),
            xpos=backend.concatenate([data.xpos for data in traj_datas], axis=0),
            xquat=backend.concatenate([data.xquat for data in traj_datas], axis=0),
            cvel=backend.concatenate([data.cvel for data in traj_datas], axis=0),
            subtree_com=backend.concatenate([data.subtree_com for data in traj_datas], axis=0),
            site_xpos=backend.concatenate([data.site_xpos for data in traj_datas], axis=0),
            site_xmat=backend.concatenate([data.site_xmat for data in traj_datas], axis=0),
            split_points=new_split_points
        )
        return new_traj_data, traj_infos[0]

    def len_trajectory(self, traj_ind):
        return self.split_points[traj_ind+1] - self.split_points[traj_ind]

    @property
    def n_trajectories(self):
        return self.split_points.shape[0] - 1

    @property
    def n_samples(self):
        return self.split_points[-1]

    @classmethod
    def get_attribute_names(cls):
        return [field.name for field in fields(cls)]

    def to_numpy(self):
        dic = flax.serialization.to_state_dict(self)
        for key, value in dic.items():
            dic[key] = np.array(value)
        return TrajectoryData(**dic)

    def to_jax(self):
        dic = flax.serialization.to_state_dict(self)
        for key, value in dic.items():
            dic[key] = jnp.array(value)
        return TrajectoryData(**dic)


def interpolate_trajectories(traj_data: TrajectoryData, traj_info: TrajectoryInfo, new_frequency: float, backend: ModuleType = jnp):
    """
    Interpolate the trajectories to a new frequency.

    Args:
        traj_data: TrajectoryData instance containing the trajectories to interpolate.
        traj_info: TrajectoryInfo instance containing the trajectory information.
        new_frequency: The frequency to interpolate the trajectories to.
        backend: Backend to use for the computation.

    Returns:
        A new instance of TrajectoryData and TrajectoryInfo containing the interpolated trajectories.

    """

    def slerp_batch(quats, times, new_times):
        """
        Perform SLERP interpolation for a batch of quaternions.

        Args:
            quats: Array of shape (T, 4) containing quaternions. (quaternions is expected to be scalar last)
            times: Array of shape (T,) containing original time points.
            new_times: Array of new time points to interpolate at.

        Returns:
            Array of shape (len(new_times), 4) containing interpolated quaternions, where the quaternion is scalar last.

        """
        # Create the Slerp object for the single trajectory
        slerp = Slerp(times, Rotation.from_quat(quats))
        # Interpolate and return the results
        return slerp(new_times).as_quat()

    def interpolate_xmat(xmats, times, new_times):
        """
        Perform interpolation for a batch of trajectories of rotation matrices.

        Args:
            xmats: Array of shape (T, N, 9) containing rotation matrices.
            times: Array of shape (T,) containing original time points.
            new_times: Array of new time points to interpolate at.

        Returns:
            Array of shape (len(new_times), N, 9) containing interpolated rotation matrices.

        """
        xmats_interpolated = []
        for i in range(traj_data_slice.site_xmat.shape[1]):
            xmat = xmats[:, i, :].reshape(-1, 3, 3)
            xquat = Rotation.from_matrix(xmat).as_quat()
            xquat_interpolated = slerp_batch(xquat, times, new_times)
            xmat_interpolated = Rotation.from_quat(xquat_interpolated).as_matrix().reshape(-1, 9)
            xmats_interpolated.append(xmat_interpolated)

        return backend.stack(xmats_interpolated, axis=1)

    old_frequency = traj_info.frequency

    new_traj_datas = []

    for i in range(traj_data.n_trajectories):

        # get the i-th trajectory
        traj_len = traj_data.len_trajectory(i)
        traj_data_slice = TrajectoryData.dynamic_slice_in_dim(traj_data, i, 0, traj_len)

        # interpolate the trajectory
        new_traj_sampling_factor = new_frequency / old_frequency
        x = backend.arange(traj_len)
        x_new = backend.linspace(0, traj_len - 1, round(traj_len * new_traj_sampling_factor), endpoint=True)

        # quaternions need SLERP interpolation
        if traj_data_slice.xquat.size > 0:
            xquat_interpolated = backend.stack([slerp_batch(traj_data_slice.xquat[:, i, :], x, x_new)
                                            for i in range(traj_data_slice.xquat.shape[1])], axis=1)
        else:
            xquat_interpolated = backend.empty(0)

        # do slerp interpolation for rotation matrices as well
        if traj_data_slice.site_xmat.size > 0:
            xmat_interpolated = interpolate_xmat(traj_data_slice.site_xmat, x, x_new)
        else:
            xmat_interpolated = backend.empty(0)

        # do slerp interpolation for free joint orientation
        qpos_free_joint_quat_ids = [i[3:] for i in traj_info.joint_name2ind_qpos.values() if len(i) > 1]
        qpos_free_joint_quat_ids_flat = [item for sublist in qpos_free_joint_quat_ids for item in sublist]
        qpos_other_ids = backend.array([i for i in range(traj_data.qpos.shape[-1])
                                    if i not in qpos_free_joint_quat_ids_flat])
        qpos = jnp.zeros((x_new.shape[0], traj_data_slice.qpos.shape[-1]))
        qpos = qpos.at[:, qpos_other_ids].set(interp1d(x, traj_data_slice.qpos[:, qpos_other_ids], kind="cubic", axis=0)(x_new))
        for quat_ids in qpos_free_joint_quat_ids:
            quat_ids = backend.array(quat_ids)
            qpos = qpos.at[:, quat_ids].set(slerp_batch(traj_data_slice.qpos[:, quat_ids], x, x_new))

        # interpolate the rest of the data
        qvel_interpolated = interp1d(x, traj_data_slice.qvel, kind="cubic", axis=0)(x_new)
        xpos_interpolated = interp1d(x, traj_data_slice.xpos, kind="cubic", axis=0)(x_new) \
            if traj_data_slice.xpos.size > 0 else jnp.empty(0)
        cvel_interpolated = interp1d(x, traj_data_slice.cvel, kind="cubic", axis=0)(x_new) \
            if traj_data_slice.cvel.size > 0 else jnp.empty(0)
        site_xpos_interpolated = interp1d(x, traj_data_slice.site_xpos, kind="cubic", axis=0)(x_new) \
            if traj_data_slice.site_xpos.size > 0 else jnp.empty(0)
        subtree_com_interpolated = interp1d(x, traj_data_slice.subtree_com, kind="cubic", axis=0)(x_new) \
            if traj_data_slice.subtree_com.size > 0 else jnp.empty(0)

        traj_data_slice = traj_data_slice.replace(
            qpos=qpos,
            qvel=qvel_interpolated,
            xpos=xpos_interpolated,
            xquat=xquat_interpolated,
            cvel=cvel_interpolated,
            site_xpos=site_xpos_interpolated,
            site_xmat=xmat_interpolated,
            subtree_com=subtree_com_interpolated,
            split_points=backend.array([0, len(x_new)]))

        new_traj_datas.append(traj_data_slice)

    traj_info = replace(traj_info, frequency=new_frequency)

    new_traj_data, traj_info = TrajectoryData.concatenate(new_traj_datas, [traj_info] * len(new_traj_datas), backend)

    return new_traj_data, traj_info


@struct.dataclass
class TrajectoryTransitions:
    """
    Data structure to store tuples of transitions observations, next_observations, actions, rewards, absorbings,
    and dones to be used for training RL algorithms.

    ..note:: Observations in this class are created using ObservationContainer.

    """

    observations: Union[jax.Array, np.ndarray]
    next_observations: Union[jax.Array, np.ndarray]
    absorbings: Union[jax.Array, np.ndarray]
    dones: Union[jax.Array, np.ndarray]

    # some datasets may not have actions and rewards (e.g., Mocap datasets)
    actions: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))
    rewards: Union[jax.Array, np.ndarray] = struct.field(default_factory=lambda: jnp.empty(0))

    def to_jnp(self):
        return jax.tree.map(lambda x: jnp.array(x), self)

    def to_np(self):
        return jax.tree.map(lambda x: np.array(x), self)

    @classmethod
    def get_attribute_names(cls):
        return [field.name for field in fields(cls)]
