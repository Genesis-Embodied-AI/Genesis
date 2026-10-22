"""
Bridge between torch autograd and the gradient tape of a differentiable scene.

Genesis tensors are plain `torch.Tensor`s, and a differentiable rollout is part of torch's own autograd graph:

- Each `scene.step()` adds a step node. It takes the token of the previous step and the inputs the entities taped for
  this step that track gradients, and returns the token of this step.
- Each tensor of a state queried after a step is the output of a read node that takes that step's token. Its gradient
  is retained, which is where the solvers read it from.

Autograd runs a node's backward only once every node that consumes its outputs has run. The backward of the node of
step t therefore runs after every read of step t and every later step, including the policy that turned a read into a
later input. It then runs the reverse pass of step t, which collects the gradients of the states read at step t, and
returns the gradients of the inputs of step t to autograd. Those flow on through whatever produced each input (a leaf,
a policy, an earlier state) within the same backward pass, so neither `retain_graph=True` nor a second traversal is
needed, and `torch.autograd.grad` works as well as `loss.backward()`.
"""

import weakref

import torch

import genesis as gs


class _StepNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scene, step, epoch, token, *inputs):
        # The scene owns the token chain, so a strong reference would make a cycle that crosses the C++ autograd graph,
        # which the garbage collector cannot traverse: the scene would never be freed.
        ctx.scene = weakref.ref(scene)
        ctx.step, ctx.epoch = step, epoch
        ctx.inputs_id = tuple(id(tensor) for tensor in inputs)
        return token.detach().clone()

    @staticmethod
    def backward(ctx, token_grad):
        scene = ctx.scene()
        if scene is None:
            gs.raise_exception("Calling backward on a scene that has been destroyed since the rollout.")
        inputs_grad = scene._reverse_step(ctx.step, ctx.epoch)
        return None, None, None, torch.zeros_like(token_grad), *(inputs_grad.get(i) for i in ctx.inputs_id)


class _ReadNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token, holder):
        # The tensor travels in a list so that autograd makes it the output itself rather than a view of an input:
        # the solver fills it in place afterwards, and users may write to it.
        return holder[0]

    @staticmethod
    def backward(ctx, grad):
        return grad.new_zeros(()), None


def grad_inputs(tgt: dict) -> list[torch.Tensor]:
    """The tensors tracking gradients in an entity's taped inputs of one step."""
    return [
        tensor
        for value in tgt.values()
        for tensor in (value.values() if isinstance(value, dict) else (value,))
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad
    ]


def step_node(scene, step: int, epoch: int, token: torch.Tensor, inputs: list[torch.Tensor]) -> torch.Tensor:
    """Add the node of step `step` to the graph and return its token."""
    return _StepNode.apply(scene, step, epoch, token, *inputs)


def scene_output(tensor: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    """Make `tensor`, a state the solver fills at the step of `token`, the output of a read node of that step."""
    if token is None:
        gs.raise_exception(
            "Only the states of a differentiable scene can track gradients. Create the scene with "
            "'SimOptions(requires_grad=True)', or pass 'requires_grad=False'."
        )
    if tensor.requires_grad:
        gs.raise_exception("A scene state is filled by the simulator, so it cannot already track gradients.")
    tensor = _ReadNode.apply(token, [tensor])
    tensor.retain_grad()
    return tensor
