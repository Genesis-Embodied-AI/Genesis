from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import taichi as ti
import torch
import torch.nn.functional as F
import math
import time
import genesis as gs
from genesis.utils.misc import tensor_to_array


# ------------------------------------------------------------------------------------
# ------------------------------------ utils -----------------------------------------
# ------------------------------------------------------------------------------------

def align_weypoints_length(
    path: torch.Tensor, mask: torch.Tensor, num_points: int  # [N, B, Dof]  # [N, B, ]
) -> torch.Tensor:
    """
    Aligns each waypoints length to the given num_points.

    Parameters
    ----------
    path: torch.Tensor
        path tensor in [N, B, Dof]
    mask: torch.Tensor
        the masking of path, indicating active waypoints
    num_points: int
        the number of the desired waypoints

    Returns
    -------
        A new 2D PyTorch tensor [num_points, B, Dof]
    """
    res = torch.zeros(path.shape[1], num_points, path.shape[-1], device=gs.device)
    for i_b in range(path.shape[1]):
        res[i_b] = torch.nn.functional.interpolate(
            path[mask[:, i_b], i_b].T.unsqueeze(0), size=num_points, mode="linear", align_corners=True
        )[0].T
    return res.transpose(1, 0)


def rrt_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the rrt connect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = tensor > 0
    mask_float = mask.float().T.unsqueeze(1)
    kernel = torch.ones(1, 1, 3, device=tensor.device)
    dilated_mask_float = F.conv1d(mask_float, kernel, padding="same")
    dilated_mask = (dilated_mask_float > 0).squeeze(1).T
    return dilated_mask


def rrt_connect_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the rrt connect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = tensor > 1
    mask_float = mask.float().T.unsqueeze(1)
    kernel = torch.ones(1, 1, 3, device=tensor.device)
    dilated_mask_float = F.conv1d(mask_float, kernel, padding="same")
    dilated_mask = (dilated_mask_float > 0).squeeze(1).T
    return dilated_mask
