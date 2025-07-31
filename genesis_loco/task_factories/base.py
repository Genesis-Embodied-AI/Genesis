from typing import Dict, Type, Any
from loco_mujoco.environments.base import LocoEnv


class TaskFactory:
    """
    A factory class for creating and registering environment instances for tasks.

    This class provides a mechanism to register task factories and create specific
    environments based on their names.

    Attributes:
        registered (Dict[str, Type["TaskFactory"]]): A dictionary mapping factory class names
                                                     to their corresponding classes.
    """

    registered: Dict[str, Type["TaskFactory"]] = dict()

    @staticmethod
    def make(env_name: str, **kwargs: Any) -> LocoEnv:
        """
        Create an environment instance.

        Args:
            env_name (str): The name of the environment to create.
            **kwargs (Any): Additional parameters required to initialize the environment.

        Returns:
            LocoEnv: An instance of the requested environment.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the `make` method.")

    @classmethod
    def get_factory_cls(cls, name: str) -> Type["TaskFactory"]:
        """
        Retrieve a registered TaskFactory class by its name.

        Args:
            name (str): The name of the TaskFactory class.

        Returns:
            Type[TaskFactory]: The factory class associated with the given name.

        Raises:
            KeyError: If the factory class is not registered.
        """
        if name not in cls.registered:
            raise KeyError(f"TaskFactory '{name}' is not registered.")
        return cls.registered[name]

    @classmethod
    def register(cls):
        """
        Register the current TaskFactory class.

        This method registers the factory class in the `registered` dictionary. It ensures
        that no duplicate factories with the same name are registered.

        Raises:
            ValueError: If the factory class is already registered.
        """
        cls_name = cls.get_name()

        if cls_name in TaskFactory.registered:
            raise ValueError(f"TaskFactory '{cls_name}' is already registered.")

        TaskFactory.registered[cls_name] = cls

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the current TaskFactory class.

        Returns:
            str: The name of the TaskFactory class.
        """
        return cls.__name__
