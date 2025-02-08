"""Utility functions for the fast_simplification package."""

import numpy as np


def ascontiguous(func):
    """A decorator that ensure that all the numpy arrays passed to the function
    are contiguous in memory and if not, apply np.ascontinguous arrays.
    """

    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[i] = np.ascontiguousarray(arg)

        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                kwargs[key] = np.ascontiguousarray(value)

        return func(*args, **kwargs)

    # Copy annotations
    wrapper.__annotations__ = func.__annotations__
    return wrapper
