"""Manager System that runs the phases of the solvers and of the coupler in the order of one simulator substep."""

from genesis.engine.core import HostAction, System


class Substepper(System):
    """Manager of one substep: the preprocess of the coupler, every pre-coupling phase, the coupling, then every
    post-coupling phase.

    A solver joins through on_substep with its rank, which is its position in the solver list of the simulator, so the
    frozen order of every phase matches the order the simulator steps the solvers in.
    """

    @System.action_collection(kind="host", call_args=("in_backward",))
    def inputs(self):
        """Host Actions that apply the external commands of one step, given whether it replays a backward window."""

    @System.action_collection(kind="host", call_args=("f",))
    def pre_couplings(self):
        """Host Actions that advance one solver up to the coupling of substep f."""

    @System.action_collection(kind="host", call_args=("f",))
    def post_couplings(self):
        """Host Actions that complete substep f of one solver once the coupling has run."""

    @System.action_collection(kind="host", call_args=("f",))
    def preprocesses(self):
        """Host Actions that prepare the coupling of substep f before any solver advances."""

    @System.action_collection(kind="host", call_args=("f",))
    def couplings(self):
        """Host Actions that exchange state between the solvers within substep f."""

    @System.protocol
    def on_substep(
        self,
        *,
        process_input: HostAction,
        pre_coupling: HostAction,
        post_coupling: HostAction,
        rank: int | None = None,
    ):
        """Join the substep loop with the three phases of one solver, calling it once from build().

        process_input  HostAction  `function(*bound_args, in_backward)`, run once per step before its substeps
        pre_coupling   HostAction  `function(*bound_args, f)`, run before the coupling of substep f
        post_coupling  HostAction  `function(*bound_args, f)`, run after the coupling of substep f
        rank           int | None  the position of the solver among the solvers of the simulator
        """
        # The same rank in all three collections keeps the solvers in one order across the phases
        self.inputs.add(process_input, rank=rank)
        self.pre_couplings.add(pre_coupling, rank=rank)
        self.post_couplings.add(post_coupling, rank=rank)

    @System.protocol
    def on_couple(self, *, preprocess: HostAction, couple: HostAction):
        """Join the substep loop with the two phases of one coupler, calling it once from build().

        preprocess  HostAction  `function(*bound_args, f)`, run before any solver advances substep f
        couple      HostAction  `function(*bound_args, f)`, run between the pre-coupling and post-coupling phases
        """
        self.preprocesses += preprocess
        self.couplings += couple

    def process_input(self, in_backward: bool) -> None:
        """Apply the external commands of one step to every solver."""
        for action in self.inputs.actions:
            action.invoke((in_backward,))

    def substep(self, f: int) -> None:
        """Advance every solver by substep f, coupling them midway."""
        for action in self.preprocesses.actions:
            action.invoke((f,))
        for action in self.pre_couplings.actions:
            action.invoke((f,))
        for action in self.couplings.actions:
            action.invoke((f,))
        for action in self.post_couplings.actions:
            action.invoke((f,))
