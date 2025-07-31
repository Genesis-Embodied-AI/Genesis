import os
from typing import Union, Dict
from omegaconf import ListConfig, DictConfig
from huggingface_hub import hf_hub_download
import yaml

import loco_mujoco
from loco_mujoco.environments.base import LocoEnv
from loco_mujoco.trajectory import Trajectory, TrajectoryHandler
from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory, extend_motion, load_robot_conf_file
from loco_mujoco.smpl.const import AMASS_LOCOMOTION_DATASETS
from loco_mujoco.datasets.humanoids.LAFAN1 import load_lafan1_trajectory
from loco_mujoco.datasets.humanoids.LAFAN1 import (LAFAN1_LOCOMOTION_DATASETS,
                                                   LAFAN1_DANCE_DATASETS, LAFAN1_ALL_DATASETS)

from .base import TaskFactory
from .dataset_confs import DefaultDatasetConf, AMASSDatasetConf, LAFAN1DatasetConf, CustomDatasetConf


class ImitationFactory(TaskFactory):
    """
    A factory class for creating imitation learning environments with arbitrary trajectories.

    Methods:
        make(env_name: str, task: str, dataset_type: str, debug: bool = False, **kwargs) -> LocoEnv:
            Creates an environment, loads a trajectory based on the task and dataset type, and returns the environment.

        get_traj_path(env_cls, dataset_type: str, task: str, debug: bool) -> str:
            Determines the path to the trajectory file based on the dataset type, task, and debug mode.
    """

    @classmethod
    def make(cls, env_name: str,
             default_dataset_conf: Union[DefaultDatasetConf, Dict, DictConfig] = None,
             amass_dataset_conf: Union[AMASSDatasetConf, Dict, DictConfig] = None,
             lafan1_dataset_conf: Union[LAFAN1DatasetConf, Dict, DictConfig] = None,
             custom_dataset_conf: Union[CustomDatasetConf, Dict, DictConfig] = None,
             terminal_state_type: str = "RootPoseTrajTerminalStateHandler",
             init_state_type: str = "TrajInitialStateHandler",
             **kwargs) -> LocoEnv:
        """
        Creates and returns an imitation learning environment given different configurations.

        Args:
            env_name (str): The name of the registered environment to create.
            default_dataset_conf (DefaultDatasetConf, optional): The configuration for the default trajectory.
            amass_dataset_conf (AMASSDatasetConf, optional): The configuration for the AMASS trajectory.
            lafan1_dataset_conf (LAFAN1DatasetConf, optional): The configuration for the LAFAN1 trajectory.
            custom_dataset_conf (CustomDatasetConf, optional): The configuration for a custom trajectory.
            terminal_state_type (str, optional): The terminal state handler to use.
                Defaults to "RootPoseTrajTerminalStateHandler".
            init_state_type (str, optional): The initial state handler to use. Defaults to "TrajInitialStateHandler".
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            LocoEnv: An instance of the requested imitation learning environment with the trajectory preloaded.

        Raises:
            ValueError: If the `dataset_type` is unknown.
        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create and return the environment
        env = env_cls(init_state_type=init_state_type, terminal_state_type=terminal_state_type, **kwargs)

        all_trajs = []

        # Load the default trajectory if available
        if default_dataset_conf is not None:
            if isinstance(default_dataset_conf, (dict, DictConfig)):
                default_dataset_conf = DefaultDatasetConf(**default_dataset_conf)
            all_trajs.append(cls.get_default_traj(env, default_dataset_conf))

        # Load the AMASS trajectory if available
        if amass_dataset_conf is not None:
            if isinstance(amass_dataset_conf, (dict, DictConfig)):
                amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)
            all_trajs.append(cls.get_amass_traj(env, amass_dataset_conf))

        # Load the LAFAN1 trajectory if available
        if lafan1_dataset_conf is not None:
            if isinstance(lafan1_dataset_conf, (dict, DictConfig)):
                lafan1_dataset_conf = LAFAN1DatasetConf(**lafan1_dataset_conf)
            all_trajs.append(cls.get_lafan1_traj(env, lafan1_dataset_conf))

        # Load the custom trajectory if available
        if custom_dataset_conf is not None:
            if isinstance(custom_dataset_conf, (dict, DictConfig)):
                custom_dataset_conf = CustomDatasetConf(**custom_dataset_conf)
            all_trajs.append(cls.get_custom_dataset(env, custom_dataset_conf))

        # concatenate trajectories
        all_trajs = Trajectory.concatenate(all_trajs)

        # add to the environment
        env.load_trajectory(traj=all_trajs, warn=False)

        return env

    @staticmethod
    def get_default_traj(env, default_dataset_conf) -> Trajectory:
        """
        Loads the default trajectory based on the dataset type, task, and debug mode.

        Args:
            env: The environment, which provides dataset paths.
            default_dataset_conf (DefaultDatasetConf): The configuration for the default trajectory.

        Returns:
            Trajectory: The default trajectories.

        Raises:
            ValueError: If the `dataset_type` is unknown.
        """
        env_name = env.__class__.__name__
        if "Mjx" in env_name:
            env_name = env_name.replace("Mjx", "")

        if isinstance(default_dataset_conf.task, str):
            default_dataset_conf.task = [default_dataset_conf.task]

        path_to_conf = loco_mujoco.PATH_TO_VARIABLES

        try:
            with open(path_to_conf, "r") as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                try:
                    path_to_convert_default_datasets = data["LOCOMUJOCO_CONVERTED_DEFAULT_PATH"]
                    os.makedirs(path_to_convert_default_datasets, exist_ok=True)
                except KeyError:
                    path_to_convert_default_datasets = None
        except FileNotFoundError:
            path_to_convert_default_datasets = None

        trajs = []
        for task in default_dataset_conf.task:

            file_path = f"{default_dataset_conf.dataset_type}/{env_name}/{task}.npz"

            traj = None
            cached_file_path = None

            # try to load the default dataset from cache
            if path_to_convert_default_datasets:
                cached_file_path = os.path.join(path_to_convert_default_datasets, file_path)
                if os.path.isfile(cached_file_path):
                    print(f"[LocoMuJoCo's Default Dataset Pipeline] INFO: Found converted dataset at: {cached_file_path}.")
                    traj = Trajectory.load(cached_file_path)

            # if the default dataset is not in the cache, download it from the hub
            if traj is None:

                filename = f"DefaultDatasets/{file_path}"

                file_path = hf_hub_download(
                    repo_id="robfiras/loco-mujoco-datasets",
                    filename=filename,
                    repo_type="dataset"
                )

                traj = Trajectory.load(file_path)

                # extend the motion to the desired length
                if not traj.data.is_complete:
                    traj = extend_motion(env_name, load_robot_conf_file(env_name).env_params, traj)

                # save  to the cache if the cache path is set
                if cached_file_path:
                    traj.save(cached_file_path)

            # pass the default trajectory through a TrajectoryHandler to interpolate it to the environment frequency
            # and to filter out or add necessary entities is needed
            default_th = TrajectoryHandler(env.model, control_dt=env.dt, traj=traj)

            trajs.append(default_th.traj)

        trajs = Trajectory.concatenate(trajs)

        return trajs

    @staticmethod
    def get_amass_traj(env, amass_dataset_conf: AMASSDatasetConf) -> Trajectory:
        """
        Determines the path to the trajectory file based on the dataset type, task, and debug mode.

        Args:
            env: The environment, which provides dataset paths.
            amass_dataset_conf (AMASSDatasetConf): The configuration for the AMASS trajectory

        Returns:
            Trajectory: The AMASS trajectories.

        Raises:
            ValueError: If the `dataset_group` is unknown.
        """

        # Determine dataset paths
        dataset_paths = []
        if amass_dataset_conf.dataset_group is not None:
            if amass_dataset_conf.dataset_group == "AMASS_LOCOMOTION_DATASETS":
                dataset_paths.extend(AMASS_LOCOMOTION_DATASETS)
            else:
                raise ValueError(f"Unknown dataset group: {amass_dataset_conf.dataset_group}")
        if amass_dataset_conf.rel_dataset_path is not None:
            dataset_paths.extend(amass_dataset_conf.rel_dataset_path
                                 if isinstance(amass_dataset_conf.rel_dataset_path, (ListConfig, list))
                                 else [amass_dataset_conf.rel_dataset_path])

        # Load AMASS Trajectory
        traj = load_retargeted_amass_trajectory(env.__class__.__name__, dataset_paths)

        # pass the default trajectory through a TrajectoryHandler to interpolate it to the environment frequency
        # and to filter out or add necessary entities is needed
        default_th = TrajectoryHandler(env.model, control_dt=env.dt, traj=traj)

        return default_th.traj

    @staticmethod
    def get_lafan1_traj(env, lafan1_dataset_conf: LAFAN1DatasetConf) -> Trajectory:
        """
        Determines the path to the trajectory file based on the dataset type, task, and debug mode.

        Args:
            env: The environment, which provides dataset paths.
            lafan1_dataset_conf (LAFAN1DatasetConf): The configuration for the LAFAN1 trajectory.

        Returns:
            Trajectory: The LAFAN1 trajectories.

        Raises:
            ValueError: If the `dataset_group` is unknown.
        """
        # Determine dataset paths
        if lafan1_dataset_conf.dataset_group:
            if lafan1_dataset_conf.dataset_group == "LAFAN1_LOCOMOTION_DATASETS":
                dataset_paths = LAFAN1_LOCOMOTION_DATASETS
            elif lafan1_dataset_conf.dataset_group == "LAFAN1_DANCE_DATASETS":
                dataset_paths = LAFAN1_DANCE_DATASETS
            elif lafan1_dataset_conf.dataset_group == "LAFAN1_ALL_DATASETS":
                dataset_paths = LAFAN1_ALL_DATASETS
            else:
                raise ValueError(f"Unknown dataset group: {lafan1_dataset_conf.dataset_group}")
        else:
            dataset_paths = lafan1_dataset_conf.dataset_name \
                if isinstance(lafan1_dataset_conf.dataset_name, (ListConfig, list)) \
                else [lafan1_dataset_conf.dataset_name]

        # Load LAFAN1 Trajectory
        traj = load_lafan1_trajectory(env.__class__.__name__, dataset_paths)

        # pass the default trajectory through a TrajectoryHandler to interpolate it to the environment frequency
        # and to filter out or add necessary entities is needed
        default_th = TrajectoryHandler(env.model, control_dt=env.dt, traj=traj)

        return default_th.traj

    @staticmethod
    def get_custom_dataset(env, custom_dataset_conf: CustomDatasetConf) -> Trajectory:
        """
        Loads the custom trajectory based on the dataset type, task, and debug mode.

        Args:
            env: The environment, which provides dataset paths.
            custom_dataset_conf (CustomDatasetConf): The configuration for the custom trajectory.

        Returns:
            Trajectory: The custom trajectories.

        """
        env_name = env.__class__.__name__
        traj = custom_dataset_conf.traj
        env_params = {}
        # # extend the motion to the desired length
        # if not traj.data.is_complete:
        #
        #     traj = extend_motion(env_name, env_params, traj)

        # pass the default trajectory through a TrajectoryHandler to interpolate it to the environment frequency
        # and to filter out or add necessary entities is needed
        default_th = TrajectoryHandler(env.model, control_dt=env.dt, traj=traj)

        return default_th.traj
