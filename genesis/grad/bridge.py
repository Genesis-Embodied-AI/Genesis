"""
Bridge between torch autograd and the gradient tape of a differentiable scene.

Genesis tensors are plain `torch.Tensor`s, and a differentiable rollout is part of torch's own autograd graph:

- Each `scene.step()` adds a step node. It takes the token of the previous step and the inputs the entities taped for
  this step that track gradients, and returns the token of this step.
- The tensors of a state queried after a step are the outputs of one read node that takes that step's token. Their
  gradients are retained, which is where the solvers read them from.

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
    def forward(ctx, sim, step, epoch, token, *inputs):
        ctx.set_materialize_grads(False)
        # The simulator owns the token chain, so a strong reference would make a cycle that crosses the C++ autograd
        # graph, which the garbage collector cannot traverse: the scene would never be freed.
        ctx.sim = weakref.ref(sim)
        ctx.step, ctx.epoch = step, epoch
        ctx.inputs_id = tuple(id(tensor) for tensor in inputs)
        return token.detach().clone()

    @staticmethod
    def backward(ctx, token_grad):
        sim = ctx.sim()
        if sim is None:
            gs.raise_exception("Differentiating a rollout whose scene was destroyed.")
        inputs_grad = sim.reverse_step(ctx.step, ctx.epoch)
        return None, None, None, None, *(inputs_grad.get(i) for i in ctx.inputs_id)


class ReadNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token, tensors):
        ctx.set_materialize_grads(False)
        # The tensors travel in a tuple so that autograd makes them the outputs themselves rather than views of inputs:
        # the solver fills them in place afterwards, and users may write to them.
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        return None, None
