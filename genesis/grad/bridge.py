"""
Bridge between torch autograd and the gradient tape of a differentiable scene.

Genesis tensors are plain `torch.Tensor`s. Gradients cross between torch's graph and the scene's tape at two places:

- Outputs. The tensors of a queried state are leaves into which torch accumulates the loss gradient. Each one carries a
  hook that, once torch has accumulated into it, schedules the scene's reverse pass at the end of that backward pass,
  so a plain `loss.backward()` differentiates through the rollout.
- Inputs. During the reverse pass, the gradient Genesis computes for a tensor handed to a tracked setter is pushed into
  that tensor's own torch graph, whatever produced it: a leaf, a policy output, or an earlier queried state.
"""

import torch

import genesis as gs


def track_scene_output(tensor: torch.Tensor, scene) -> None:
    """Make `tensor`, a leaf of a state queried from `scene`, start the scene's reverse pass when it gets a gradient."""
    if not tensor.is_leaf:
        gs.raise_exception("Only leaf tensors can be tracked as scene outputs.")
    tensor.register_post_accumulate_grad_hook(lambda _: scene._queue_backward())


def backward_from_qd(tensor: torch.Tensor, qd_kernel, *args) -> None:
    """Fill the gradient of `tensor` with `qd_kernel(*args, grad)` and back-propagate it through `tensor`'s graph."""
    grad = torch.zeros_like(tensor, requires_grad=False)
    qd_kernel(*args, grad)
    torch.autograd.backward(tensor, grad, retain_graph=True)
