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
            repr_str += f"{link_i._repr_type()}\n"
        repr_str = repr_str[:-1] + "]"
        return repr_str


class QueriedStates:
    """
    A dict of queried states.
    """

    def __init__(self):
        self.states = dict()

    def append(self, state):
        if state.s_global not in self.states:
            self.states[state.s_global] = StateList([state])
        else:
            self.states[state.s_global].append(state)

    def clear(self):
        self.states.clear()

    def __contains__(self, key):
        return key in self.states

    def __getitem__(self, key):
        return self.states[key]

    def __repr__(self):
        return f"{brief(self)}\n" f"states: {brief(self.states)}"
