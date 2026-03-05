import genesis as gs

from .kinematic import Kinematic


class Rigid(Kinematic):
    """
    The Rigid class represents a material used in rigid body simulation.

    Note
    ----
    This class is intended for use with the rigid solver and provides parameters
    relevant to physical interactions such as friction, density, and signed distance fields (SDFs).

    Parameters
    ----------
        rho : float, optional
            The density of the material used to compute mass. Default is 200.0.
        friction : float, optional
            Friction coefficient within the rigid solver. If None, a default of 1.0 may be used or parsed from file.
        needs_coup : bool, optional
            Whether the material participates in coupling with other solvers. Default is True.
        coup_friction : float, optional
            Friction used during coupling. Must be non-negative. Default is 0.1.
        coup_softness : float, optional
            Softness of coupling interaction. Must be non-negative. Default is 0.002.
        coup_restitution : float, optional
            Restitution coefficient in collision coupling. Should be between 0 and 1. Default is 0.0.
        sdf_cell_size : float, optional
            Cell size in SDF grid in meters. Defines grid resolution. Default is 0.005.
        sdf_min_res : int, optional
            Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
        sdf_max_res : int, optional
            Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
        gravity_compensation : float, optional
            Compensation factor for gravity. 1.0 cancels gravity. Default is 0.
        coup_type : str or None, optional
            Coupling mode for this entity. Only used by the IPC coupler. Requires ``needs_coup=True``.
            If None, auto-selected based on entity type: ``'external_articulation'`` for fixed-base
            articulated robots, ``'two_way_soft_constraint'`` for floating-base robots, and
            ``'ipc_only'`` for non-articulated objects. Valid values:
              - 'two_way_soft_constraint': Two-way soft coupling.
              - 'external_articulation': Joint-level coupling for articulated bodies. Joint positions will be coupled at
                the DOF level.
              - 'ipc_only': IPC controls entity, transforms copied to Genesis (one-way). Only supported by rigid
                non-articulated objects.
            Default is None.
        coup_links : tuple of str or None, optional
            Tuple of link names to include in coupling. When set, only the named links participate
            in coupling; other links are excluded. Only supported with needs_coup=True and
            ``two_way_soft_constraint`` type in IPC. Default is None.
        enable_coup_collision : bool, optional
            Whether coupler collision is enabled for this entity's links. Only used by the IPC coupler.
            Unlike ``needs_coup=False`` (which removes the entity from the coupler entirely), setting this to
            False keeps the entity in the coupler for coupling forces but disables contact response. Default is True.
        coup_collision_links : tuple of str or None, optional
            Tuple of link names whose geoms participate in coupler collision. Only used by the IPC coupler.
            Only effective when ``enable_coup_collision=True``. If None, all coupled links have collision.
            When set, only the named links get coupler collision; other links are marked no-collision.
            Default is None.
        contact_resistance : float or None, optional
            IPC coupling contact resistance/stiffness override for this entity. ``None`` means use
            ``IPCCouplerOptions.contact_resistance``. Default is None.
    """

    def __init__(
        self,
        rho=200.0,
        friction=None,
        needs_coup=True,
        coup_friction=0.1,
        coup_softness=0.002,
        coup_restitution=0.0,
        sdf_cell_size=0.005,
        sdf_min_res=32,
        sdf_max_res=128,
        gravity_compensation=0.0,
        coup_type=None,
        coup_links=None,
        enable_coup_collision=True,
        coup_collision_links=None,
        contact_resistance=None,
    ):
        super().__init__()

        if coup_type is not None:
            if not needs_coup:
                gs.raise_exception(
                    "`coup_type` is only supported with needs_coup=True. "
                    f"Got needs_coup={needs_coup}, coup_type='{coup_type}'."
                )
            if coup_type not in ("two_way_soft_constraint", "external_articulation", "ipc_only"):
                gs.raise_exception(
                    f"`coup_type` must be one of None, 'two_way_soft_constraint', "
                    f"'external_articulation', or 'ipc_only', got '{coup_type}'."
                )

        if coup_links is not None and (not needs_coup or coup_type not in (None, "two_way_soft_constraint")):
            gs.raise_exception(
                "`coup_links` is only supported with needs_coup=True and 'two_way_soft_constraint' type in IPC. "
                f"Got needs_coup={needs_coup}, coup_type='{coup_type}'."
            )

        if coup_collision_links is not None and not enable_coup_collision:
            gs.raise_exception(
                "`coup_collision_links` is only effective when `enable_coup_collision=True`. "
                "Set `enable_coup_collision=False` to disable collision for all links."
            )

        if friction is not None:
            if friction < 1e-2 or friction > 5.0:
                gs.raise_exception("`friction` must be in the range [1e-2, 5.0] for simulation stability.")

        if coup_friction < 0:
            gs.raise_exception("`coup_friction` must be non-negative.")

        if coup_softness < 0:
            gs.raise_exception("`coup_softness` must be non-negative.")

        if contact_resistance is not None and contact_resistance <= 0:
            gs.raise_exception("`contact_resistance` must be strictly positive.")

        if coup_restitution < 0 or coup_restitution > 1:
            gs.raise_exception("`coup_restitution` must be in the range [0, 1].")

        if coup_restitution != 0:
            gs.logger.warning("Non-zero `coup_restitution` could lead to instability. Use with caution.")

        if sdf_min_res < 16:
            gs.raise_exception("`sdf_min_res` must be at least 16.")

        if sdf_min_res > sdf_max_res:
            gs.raise_exception("`sdf_min_res` must be smaller than or equal to `sdf_max_res`.")

        # ipc_only entities have their dynamics fully controlled by IPC (gravity + collision).
        # Genesis gravity must be disabled to avoid double-counting.
        if coup_type == "ipc_only":
            if abs(gravity_compensation) > gs.EPS:
                gs.raise_exception("User-specified `gravity_compensation` not supported with coup_type='ipc_only'.")

        self._friction = float(friction) if friction is not None else None
        self._needs_coup = bool(needs_coup)
        self._coup_friction = float(coup_friction)
        self._coup_softness = float(coup_softness)
        self._coup_restitution = float(coup_restitution)
        self._sdf_cell_size = float(sdf_cell_size)
        self._sdf_min_res = int(sdf_min_res)
        self._sdf_max_res = int(sdf_max_res)
        self._rho = float(rho)
        self._gravity_compensation = float(gravity_compensation)
        self._coup_type = coup_type
        self._coup_links = tuple(coup_links) if coup_links is not None else None
        self._enable_coup_collision = bool(enable_coup_collision)
        self._coup_collision_links = tuple(coup_collision_links) if coup_collision_links is not None else None
        self._contact_resistance = float(contact_resistance) if contact_resistance is not None else None

    @property
    def gravity_compensation(self) -> float:
        """Gravity compensation factor. 1.0 cancels gravity."""
        return self._gravity_compensation

    @property
    def friction(self) -> float | None:
        """Friction coefficient used within the rigid solver."""
        return self._friction

    @property
    def needs_coup(self) -> bool:
        """Whether this material requires solver coupling."""
        return self._needs_coup

    @property
    def coup_friction(self) -> float:
        """Friction coefficient used in coupling interactions."""
        return self._coup_friction

    @property
    def coup_softness(self) -> float:
        """Softness parameter controlling the influence range of coupling."""
        return self._coup_softness

    @property
    def coup_restitution(self) -> float:
        """Restitution coefficient used during contact in coupling."""
        return self._coup_restitution

    @property
    def contact_resistance(self) -> float | None:
        """IPC coupling contact resistance/stiffness override, or None for coupler default."""
        return self._contact_resistance

    @property
    def sdf_cell_size(self) -> float:
        """Size of each SDF grid cell in meters."""
        return self._sdf_cell_size

    @property
    def sdf_min_res(self) -> int:
        """Minimum allowed resolution for the SDF grid."""
        return self._sdf_min_res

    @property
    def sdf_max_res(self) -> int:
        """Maximum allowed resolution for the SDF grid."""
        return self._sdf_max_res

    @property
    def rho(self) -> float:
        """Density of the rigid material."""
        return self._rho

    @property
    def coup_type(self) -> str | None:
        """IPC coupling mode for this entity."""
        return self._coup_type

    @property
    def coup_links(self) -> tuple[str, ...] | None:
        """Tuple of link names to include in coupling."""
        return self._coup_links

    @property
    def enable_coup_collision(self) -> bool:
        """Whether IPC collision is enabled for this entity's links."""
        return self._enable_coup_collision

    @property
    def coup_collision_links(self) -> tuple[str, ...] | None:
        """Tuple of link names whose geoms participate in IPC collision. None = all coupled links."""
        return self._coup_collision_links
