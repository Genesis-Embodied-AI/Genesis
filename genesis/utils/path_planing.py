import numpy as np
import taichi as ti
import torch

import genesis as gs


def move_padding_to_tail_vectorized(tensor: torch.Tensor) -> torch.Tensor:
    """
    Moves leading zero-padding to the tail for each column in a 2D tensor.
    The tail is padded with the last value of the original column.

    Parameters
    ----------
    tensor: torch.Tensor
        A 2D PyTorch tensor of shape [N, B] with integer types.

    Returns 
    -------
        A new 2D PyTorch tensor with the padding transformed.
    """
    # Get the dimensions of the input tensor
    n_dim, b_dim = tensor.shape

    # Create a boolean mask of non-zero elements
    non_zero_mask = (tensor != 0)

    # Find the index of the first non-zero element in each column.
    # .argmax() finds the first 'True' (or 1) along dimension 0.
    first_nonzero_indices = non_zero_mask.int().argmax(dim=0)

    # Handle the edge case of all-zero columns. For these columns, argmax
    # returns 0, which is incorrect for our logic. We find these columns
    # and set their first non-zero index to N, effectively treating them
    # as having no content and full padding.
    is_all_zero_column = ~non_zero_mask.any(dim=0)
    first_nonzero_indices[is_all_zero_column] = n_dim

    # Calculate the length of the content for each column
    content_len = n_dim - first_nonzero_indices # Shape: [B]

    # Create a tensor representing the row indices from 0 to N-1
    # Shape will be broadcast from [N, 1] to [N, B] in subsequent operations.
    arange_n = torch.arange(n_dim, device=tensor.device).unsqueeze(1)

    # Create the indices for the content part of each column.
    # This effectively "rolls" the data up by `first_nonzero_indices`.
    content_indices = arange_n + first_nonzero_indices # Shape: [N, B]

    # Create a boolean mask to identify which elements fall into the
    # new tail padding section for each column.
    is_padding_part = arange_n >= content_len # Shape: [N, B]

    # The index for all padding values is the last row (N-1).
    # This will be broadcast from [1, B] to [N, B] by torch.where.
    last_row_index = torch.full((1, b_dim), n_dim - 1, device=tensor.device)

    # Use the `is_padding_part` mask to choose the final indices.
    # If it's a content part, use the rolled `content_indices`.
    # If it's a padding part, use the `last_row_index`.
    final_indices = torch.where(is_padding_part, last_row_index, content_indices)

    # Use torch.gather to select elements from the original tensor based on our
    # computed `final_indices`. This performs the entire transformation in a
    # single, highly optimized operation.
    result_tensor = torch.gather(tensor, 0, final_indices)

    return result_tensor