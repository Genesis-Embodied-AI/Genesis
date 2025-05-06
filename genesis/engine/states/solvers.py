import genesis as gs
from genesis.repr_base import RBC


class SimState(RBC):
    """
    Dynamic state queried from a Scene's Simulator.
    """

    def __init__(
        self,
        scene,
        s_global,
        f_local,
        solvers,
    ):
        self._scene = scene
        self._s_global = s_global
        self._solvers_state = list()
        for solver in solvers:
            self._solvers_state.append(solver.get_state(f_local))

    def serializable(self):
        self.scene = None

        for solver_state in self._solvers_state:
            if solver_state is not None:
                solver_state.serializable()

    @property
    def scene(self):
        return self._scene

    @property
    def s_global(self):
        return self._s_global

    @property
    def solvers_state(self):
        return self._solvers_state

    def __iter__(self):
        return iter(self._solvers_state)


class RigidSolverState:
    """
    Dynamic state queried from a RigidSolver.
    """

    def __init__(self, scene):
        self.scene = scene
        args = {
            "dtype": float,
            "requires_grad": scene.requires_grad,
            "scene": self.scene,
        }
        self.qpos = gs.zeros(scene.sim.rigid_solver._batch_shape(scene.sim.rigid_solver.n_qs, True), **args)
        self.dofs_vel = gs.zeros(scene.sim.rigid_solver._batch_shape(scene.sim.rigid_solver.n_dofs, True), **args)
        n_links = scene.sim.rigid_solver.n_links
        n_geoms = scene.sim.rigid_solver.n_geoms
        self.links_pos = gs.zeros(scene.sim.rigid_solver._batch_shape((n_links, 3), True), **args)
        self.links_quat = gs.zeros(scene.sim.rigid_solver._batch_shape((n_links, 4), True), **args)
        self.i_pos_shift = gs.zeros(scene.sim.rigid_solver._batch_shape((n_links, 3), True), **args)
        self.mass_shift = gs.zeros(scene.sim.rigid_solver._batch_shape(n_links, True), **args)
        self.friction_ratio = gs.ones(scene.sim.rigid_solver._batch_shape(n_geoms, True), **args)

    def serializable(self):
        self.scene = None
        self.qpos = self.qpos.detach()
        self.dofs_vel = self.dofs_vel.detach()
        self.links_pos = self.links_pos.detach()
        self.links_quat = self.links_quat.detach()
        self.i_pos_shift = self.i_pos_shift.detach()
        self.mass_shift = self.mass_shift.detach()
        self.friction_ratio = self.friction_ratio.detach()


class AvatarSolverState:
    """
    Dynamic state queried from a AvatarSolver.
    """

    def __init__(self, scene):
        self.scene = scene
        args = {
            "dtype": float,
            "requires_grad": scene.requires_grad,
            "scene": self.scene,
        }
        self.qpos = gs.zeros(scene.sim.avatar_solver._batch_shape(scene.sim.avatar_solver.n_qs, True), **args)
        self.dofs_vel = gs.zeros(scene.sim.avatar_solver._batch_shape(scene.sim.avatar_solver.n_dofs, True), **args)

        n_links = scene.sim.avatar_solver.n_links
        self.links_pos = gs.zeros(scene.sim.avatar_solver._batch_shape((n_links, 3), True), **args)
        self.links_quat = gs.zeros(scene.sim.avatar_solver._batch_shape((n_links, 4), True), **args)

    def serializable(self):
        self.scene = None
        self.qpos = self.qpos.detach()
        self.dofs_vel = self.dofs_vel.detach()
        self.links_pos = self.links_pos.detach()
        self.links_quat = self.links_quat.detach()


class ToolSolverState:
    """
    Dynamic state queried from a RigidSolver.
    """

    def __init__(self, scene):
        self.scene = scene
        self.entities = []

    def serializable(self):
        self.scene = None

        for entity_state in self.entities:
            entity_state.serializable()

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        return self.entities[index]

    # def __repr__(self):
    #     return f'{_repr(self)}\n' \
    #            f'entities : {_repr(self.entities)}'


class MPMSolverState(RBC):
    """
    Dynamic state queried from a MPMSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3), **args)
        self._vel = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3), **args)
        self._C = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3, 3), **args)
        self._F = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3, 3), **args)
        self._Jp = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles), **args)
        args["dtype"] = int
        args["requires_grad"] = False
        self._active = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles), **args)

    def serializable(self):
        self._scene = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._C = self._C.detach()
        self._F = self._F.detach()
        self._Jp = self._Jp.detach()
        self._active = self._active.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def active(self):
        return self._active


class SPHSolverState:
    """
    Dynamic state queried from a SPHSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.sph_solver.n_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, scene.sim.sph_solver.n_particles, 3), **args)
        args["dtype"] = int
        args["requires_grad"] = False
        self._active = gs.zeros((self._scene.sim._B, scene.sim.sph_solver.n_particles), **args)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class PBDSolverState:
    """
    Dynamic state queried from a PBDSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.pbd_solver.n_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, scene.sim.pbd_solver.n_particles, 3), **args)
        args["dtype"] = int
        args["requires_grad"] = False
        self._free = gs.zeros((self._scene.sim._B, scene.sim.pbd_solver.n_particles), **args)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def free(self):
        return self._free

    # def __repr__(self):
    #     return f'{_repr(self)}\n' \
    #            f'scene : {_repr(self.scene)}\n' \
    #            f'pos   : {_repr(self.pos)}\n' \
    #            f'vel   : {_repr(self.vel)}\n'


class FEMSolverState:
    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_vertices, 3), **args)
        self._vel = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_vertices, 3), **args)
        args["dtype"] = int
        args["requires_grad"] = False
        self._active = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_elements), **args)

    def serializable(self):
        self._scene = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._active = self._active.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active
