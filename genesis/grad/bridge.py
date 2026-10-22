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

The token only orders the nodes, and its gradient is always zero. Autograd runs a node on the worker thread of the device
of the gradients it receives, and the queues of different devices concurrently. The token and its gradients therefore
live on `gs.device`, so the reverse pass of every step runs on the thread that runs the torch operations of the policy
around it: two threads issuing work to the same device stream concurrently is unsafe on Metal.
"""

import weakref

import torch

import genesis as gs


class StepNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sim, step, epoch, token, *inputs):
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
            gs.raise_exception("Calling backward on a scene that has been destroyed since the rollout.")
        inputs_grad = sim.reverse_step(ctx.step, ctx.epoch)
        return None, None, None, token_grad, *(inputs_grad.get(i) for i in ctx.inputs_id)


class ReadNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token, tensors):
        # The tensors travel in a tuple so that autograd makes them the outputs themselves rather than views of inputs:
        # the solver fills them in place afterwards, and users may write to them.
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        return grads[0].new_zeros(()), None


class QueriedStates:
    """
    The states queried during the current rollout of a differentiable scene, grouped by the step they were read at.

    Registering a state makes the tensors it declares in 'grad_tensors' differentiable (see 'Simulator.track_state'),
    and 'collect_output_grads' later hands their gradients to the solver or entity that registered it. A scene that is
    not differentiable keeps nothing, since nothing will read the gradients, and neither does a query made while torch
    records no gradient ('torch.no_grad', 'torch.inference_mode'), whose state is not differentiable.
    """

    def __init__(self, sim):
        self._sim = sim
        self._states_by_step: dict[int, list] = {}

    def append(self, state):
        if not (self._sim.requires_grad and torch.is_grad_enabled()):
            return
        self._sim.track_state(*state.grad_tensors)
        self._states_by_step.setdefault(self._sim.cur_step_global, []).append(state)

    def get(self, s_global):
        """Return the states registered at the given global step, if any."""
        return self._states_by_step.get(s_global, ())

    def clear(self):
        self._states_by_step.clear()
