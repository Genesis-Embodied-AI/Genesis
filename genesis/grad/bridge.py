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

The token only orders the nodes, so its gradient is never materialized. It lives on CPU: autograd runs a node on the
device of the gradients it receives, so the reverse pass of every step runs on the thread calling backward.
"""

import weakref

import torch

import genesis as gs


class StepNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scene, step, epoch, token, *inputs):
        ctx.set_materialize_grads(False)
        # The scene owns the token chain, so a strong reference would make a cycle that crosses the C++ autograd
        # graph, which the garbage collector cannot traverse: the scene would never be freed.
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
        return None, None, None, None, *(inputs_grad.get(i) for i in ctx.inputs_id)


class ReadNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token, holder):
        ctx.set_materialize_grads(False)
        # The tensor travels in a list so that autograd makes it the output itself rather than a view of an input:
        # the solver fills it in place afterwards, and users may write to it.
        return holder[0]

    @staticmethod
    def backward(ctx, grad):
        return None, None
