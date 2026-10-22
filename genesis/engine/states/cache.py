import torch

from genesis.utils.repr import brief


class StateList(list):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return StateList(result)
        else:
            return result

    def __repr__(self):
        repr_str = "["
        for link_i in self:
            repr_str += f"{link_i.__repr_name__()}\n"
        repr_str = repr_str[:-1] + "]"
        return repr_str


class QueriedStates:
    """
    The states queried during the current rollout of a differentiable scene, grouped by the step they were read at.

    Registering a state makes the tensors it declares in 'grad_tensors' differentiable (see 'Simulator.track_state'),
    and 'collect_output_grads' later hands their gradients to the solver or entity that registered it. A scene that is
    not differentiable keeps nothing, since nothing will read the gradients, and neither does a query made while torch
    records no gradient ('torch.no_grad', 'torch.inference_mode'): such a state is not differentiable, so the solvers
    that return the state registered at the current step must never hand it to a later query.
    """

    def __init__(self, sim):
        self._sim = sim
        self.states = dict()

    def append(self, state):
        if not (self._sim.requires_grad and torch.is_grad_enabled()):
            return
        self._sim.track_state(*state.grad_tensors)
        self.states.setdefault(self._sim.cur_step_global, StateList()).append(state)

    def get(self, s_global):
        """Return the states registered at the given global step, if any."""
        return self.states.get(s_global, ())

    def clear(self):
        self.states.clear()

    def __contains__(self, key):
        return key in self.states

    def __getitem__(self, key):
        return self.states[key]

    def __repr__(self):
        return f"{brief(self)}\nstates: {brief(self.states)}"
