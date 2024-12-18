import numpy as np
import torch

import genesis as gs


def brief(x):
    if hasattr(x, "_repr_brief"):
        return x._repr_brief()

    elif isinstance(x, (int, float, dict, bool, list, tuple, np.integer, np.floating)):
        return f"{_repr_type(x)}: {x}"

    elif isinstance(x, str):
        return f"{_repr_type(x)}: '{x}'"

    elif isinstance(x, (np.ndarray, torch.Tensor)):
        if np.prod(x.shape) <= 20:
            return f"{_repr_type(x)}: {x.__repr__()}"
        else:
            return f"{_repr_type(x)}, shape: {x.shape}"

    # elif isinstance(x, (gs.IntEnum, gs.UID)):
    #     return x.__repr__()

    elif x is None:
        return "None"

    else:
        return _repr_type(x)


def _repr_type(x):
    """
    Only used for non-genesis object by `brief()`.
    To convert <class 'classname'> into <classname>.
    """
    if isinstance(x, type):
        raw_class_name = str(x)
    else:
        raw_class_name = str(x.__class__)
    full_name = f"<{' '.join(raw_class_name.split(' ')[1:])[1:-2]}>"
    return full_name
