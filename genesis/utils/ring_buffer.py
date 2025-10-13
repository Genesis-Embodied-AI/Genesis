import torch

import genesis as gs


class TensorRingBuffer:
    """
    A helper class for storing a buffer of `torch.Tensor`s without allocating new tensors.

    Parameters
    ----------
    N : int
        The number of tensors to store.
    shape : tuple[int, ...]
        The shape of the tensors to store.
    dtype : torch.dtype
        The dtype of the tensors to store.
    buffer : torch.Tensor | None, optional
        The buffer tensor where all the data is stored. If not provided, a new tensor is allocated.
    idx : torch.Tensor, optional
        The index reference to the most recently updated position in the ring buffer as a mutable 0D torch.Tensor of
        integer dtype. If not provided, it is initialized to -1.
    """

    def __init__(
        self,
        N: int,
        shape: tuple[int, ...],
        dtype=torch.float32,
        buffer: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
    ):
        if buffer is None:
            self.buffer = torch.empty((N, *shape), dtype=dtype, device=gs.device)
        else:
            assert buffer.shape == (N, *shape)
            self.buffer = buffer
        self.N = N
        if idx is None:
            self._idx = torch.tensor(-1, dtype=torch.int64, device=gs.device)
        else:  # torch.Tensor
            assert idx.ndim == 0 and idx.dtype in (torch.int32, torch.int64)
            self._idx = idx.to(device=gs.device)
            assert self._idx is idx

    def append(self, tensor: torch.Tensor):
        """
        Copy the tensor into the next position of the ring buffer, and advance the index pointer.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to copy into the ring buffer.
        """
        self._idx[()] = (self._idx + 1) % self.N
        self.buffer[self._idx].copy_(tensor)

    def at(
        self, idx: int | torch.Tensor, *others_idx: int | slice | torch.Tensor, copy: bool | None = None
    ) -> torch.Tensor:
        """
        Get the value of the tensor at the given index.

        Parameters
        ----------
        idx : int | torch.Tensor
            Index of the element to get from most recent to least recent (that has not been discarded yet).
            Passing a 1D tensor for advanced (aka fancy) indexing is supported, but this is requiring allocating fresh
            memory instead of returning a view, which is less efficient.
        others_idx : int | slice | torch.Tensor, optional
            Index of the elements to extract from the selected tensor. In case of advanced indexing, this is equivalent
            but significantly more efficient than doing this extraction in a latter stage.
        copy: bool | None, optional
            If `None`, then memory will be allocated only if necessary. If `True`, then memory will be allocated
            systematically instead of returning a view. If `False`, then allocating memory is forbidden and will raise
            an exception if returning a view is impossible.
        """
        rel_idx = (self._idx - idx) % self.N
        assert len(others_idx) < self.buffer.ndim
        tensor = self.buffer[(rel_idx, *others_idx)]
        if tensor.untyped_storage().data_ptr() == self.buffer.untyped_storage().data_ptr():
            if copy:
                tensor = tensor.clone()
        elif copy == False:
            gs.raise_exception("Allocating memory is necessary but 'copy=False'.")
        return tensor

    def get(self, idx: int) -> torch.Tensor:
        """
        Get a clone of the tensor at the given index.

        Parameters
        ----------
        idx : int
            Index of the element to get from most recent to least recent (that has not been discarded yet).
        """
        return self.at(idx, copy=True)

    def clone(self) -> "TensorRingBuffer":
        return TensorRingBuffer(
            self.N,
            self.buffer.shape[1:],
            dtype=self.buffer.dtype,
            buffer=self.buffer.clone(),
            idx=self._idx,
        )

    def __getitem__(self, key: int | slice | tuple) -> "TensorRingBuffer":
        """
        Enable slicing of the tensor ring buffer.

        Parameters
        ----------
        key : int | slice | tuple
            Slice object (e.g., 3:6) or integer index or tuple of indices

        Returns
        -------
        TensorRingBuffer
            A new ring buffer containing a view of the sliced data
        """
        if isinstance(key, int):
            sliced_buffer = self.buffer[:, key : key + 1]
        elif isinstance(key, slice):
            sliced_buffer = self.buffer[:, key]
        elif isinstance(key, tuple):
            indexes = (slice(None),) + key
            sliced_buffer = self.buffer[indexes]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        return TensorRingBuffer(
            self.N,
            sliced_buffer.shape[1:],
            dtype=sliced_buffer.dtype,
            buffer=sliced_buffer,
            idx=self._idx,
        )
