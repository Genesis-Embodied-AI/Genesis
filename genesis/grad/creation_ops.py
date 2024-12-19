import torch

import genesis as gs

from .tensor import Tensor

_torch_creation_ops = [
    torch.tensor,
    torch.asarray,
    torch.as_tensor,
    torch.as_strided,
    torch.from_numpy,
    torch.zeros,
    torch.zeros_like,
    torch.ones,
    torch.ones_like,
    torch.arange,
    torch.range,
    torch.linspace,
    torch.logspace,
    torch.eye,
    torch.empty,
    torch.empty_like,
    torch.empty_strided,
    torch.full,
    torch.full_like,
    torch.rand,
    torch.rand_like,
    torch.randn,
    torch.randn_like,
    torch.randint,
    torch.randint_like,
    torch.randperm,
]

_us_creation_op_template = """
def {op.__name__}(*args, **kwargs):
    return torch_op_wrapper(torch.{op.__name__}, *args, **kwargs)

{op.__name__}.__doc__ = "This is the genesis wrapper of torch.{op.__name__}()."
"""


def _is_float(torch_tensor):
    return torch_tensor.dtype in [torch.float32, torch.float64]


def _is_int(torch_tensor):
    return torch_tensor.dtype in [torch.int32, torch.int64]


def torch_op_wrapper(torch_op, *args, dtype=None, requires_grad=False, scene=None, **kwargs):
    if "device" in kwargs:
        gs.raise_exception("Device selection not supported. All genesis tensors are on GPU.")

    if not gs._initialized:
        gs.raise_exception("Genesis not initialized yet.")

    if torch_op is torch.from_numpy:
        torch_tensor = torch_op(*args)
    else:
        torch_tensor = torch_op(*args, **kwargs)

    gs_tensor = from_torch(torch_tensor, dtype, requires_grad, detach=True, scene=scene)
    return gs_tensor


def from_torch(torch_tensor, dtype=None, requires_grad=False, detach=True, scene=None):
    """
    By default, detach is True, meaning that this function returns a new leaf tensor which is not connected to torch_tensor's computation gragh.
    """
    if dtype is None:
        if _is_float(torch_tensor):
            dtype = gs.tc_float
        elif _is_int(torch_tensor):
            dtype = gs.tc_int
        else:
            dtype = torch_tensor.dtype
    elif dtype is float:
        dtype = gs.tc_float
    elif dtype is int:
        dtype = gs.tc_int
    else:
        gs.raise_exception("Supported dtype: [None, int, float]")

    if torch_tensor.requires_grad and (not detach) and (not requires_grad):
        gs.logger.warning(
            "The parent torch tensor requires grad and detach is set to False. Ignoring requires_grad=False."
        )
        requires_grad = True

    gs_tensor = Tensor(torch_tensor.to(device=gs.device, dtype=dtype), scene=scene).clone()

    if detach:
        gs_tensor = gs_tensor.detach(sceneless=False)

    if requires_grad:
        gs_tensor = gs_tensor.requires_grad_()

    return gs_tensor


_raw_map = {8: r"\b", 7: r"\a", 12: r"\f", 10: r"\n", 13: r"\r", 9: r"\t", 11: r"\v"}


def _convert_to_raw_str(s):
    return r"".join([_raw_map.get(ord(c), c) for c in s])


for op in _torch_creation_ops:
    exec(eval(_convert_to_raw_str(f"f'{_us_creation_op_template}'")))
