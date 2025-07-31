from .base import TaskFactory
from loco_mujoco.environments.base import LocoEnv


class RLFactory(TaskFactory):
    """
    A factory class to create reinforcement learning (RL) environments.

    Methods:
        make(env: str, **kwargs) -> LocoEnv:
            Creates and returns an RL environment based on the specified environment name.
    """

    @staticmethod
    def make(env_name: str,
             init_state_type: str = "DefaultInitialStateHandler",
             terminal_state_type: str = "HeightBasedTerminalStateHandler",
             goal_type: str = "GoalRandomRootVelocity",
             reward_type: str = "TargetVelocityGoalReward",
             **kwargs) -> LocoEnv:
        """
        Creates and returns an RL environment based on the specified environment name.

        Args:
            env_name (str): The name of the registered environment to create.
            init_state_type (str, optional): The initial state handler to use. Defaults to "DefaultInitialStateHandler".
            terminal_state_type (str, optional): The terminal state handler to use.
                Defaults to "HeightBasedTerminalStateHandler".
            goal_type (str, optional): The goal handler to use. Defaults to "GoalRandomRootVelocity".
            reward_type (str, optional): The reward handler to use. Defaults to "TargetVelocityGoalReward".
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            LocoEnv: An instance of the requested RL environment.

        Raises:
            KeyError: If the specified environment is not registered in `Mujoco.registered_envs`.
        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create and return the environment
        return env_cls(init_state_type=init_state_type,
                       terminal_state_type=terminal_state_type,
                       goal_type=goal_type,
                       reward_type=reward_type,
                       **kwargs)
