"""
base.py
-------------

The base class for `Visual` objects
"""

import abc
from ..util import ABC


class Visuals(ABC):
    """
    Parent of Visual classes.
    """

    @abc.abstractproperty
    def kind(self):
        pass

    @abc.abstractmethod
    def update_vertices(self, mask):
        pass

    @abc.abstractmethod
    def update_faces(self, mask):
        pass

    @abc.abstractmethod
    def concatenate(self, other):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    def __add__(self, other):
        """
        Concatenate two ColorVisuals objects into a single object.

        Parameters
        -----------
        other : Visuals
          Other visual to concatenate

        Returns
        -----------
        result : Visuals
          Object containing information from current
          object and other in the order (self, other)
        """
        return self.concatenate(other)
