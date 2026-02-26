"""
Cloth material for IPC-based cloth simulation.

This material is used with FEMEntity and IPCCoupler for shell/membrane simulation.
"""

from .base import Base


class Cloth(Base):
    """
    Cloth material for thin shell/membrane simulation using IPC.

    This material is designed for cloth, fabric, and other thin flexible materials.
    It uses shell-based FEM formulation (NeoHookeanShell) in the IPC backend.

    When used with FEMEntity, it signals to IPCCoupler that this entity should be
    treated as a 2D shell (cloth) rather than a 3D volumetric FEM object.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa), controlling stiffness. Default is 1e4 (10 kPa).
    nu : float, optional
        Poisson's ratio, describing volume change under stress.
        Default is 0.49 (nearly incompressible).
    rho : float, optional
        Material density (kg/m³). Default is 200 (typical fabric).
    thickness : float, optional
        Shell thickness (m). Default is 0.001 (1mm).
    bending_stiffness : float, optional
        Bending resistance coefficient. If None, no bending resistance.
        Default is None.
    model : str, optional
        FEM material model (not used for cloth, kept for compatibility).
        Default is "stable_neohookean".
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the IPC coupler
        global default. Default is None.

    Notes
    -----
    - Only works with IPCCoupler enabled
    - Requires GPU backend
    - Only accepts surface mesh morphs (Mesh, etc.)
    - Uses FEMEntity infrastructure but simulated as 2D shell in IPC

    Examples
    --------
    >>> cloth = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="cloth.obj"),
    ...     material=gs.materials.FEM.Cloth(
    ...         E=10e3, nu=0.49, rho=200,
    ...         thickness=0.001, bending_stiffness=10.0
    ...     ),
    ... )
    """

    def __init__(
        self,
        E=1e4,  # Young's modulus (Pa)
        nu=0.49,  # Poisson's ratio
        rho=200.0,  # Density (kg/m³)
        thickness=0.001,  # Shell thickness (m)
        bending_stiffness=None,  # Optional bending stiffness
        model="stable_neohookean",  # FEM model (unused for cloth)
        friction_mu=0.1,
        contact_resistance=None,
    ):
        # Call FEM base constructor
        super().__init__(E=E, nu=nu, rho=rho, friction_mu=friction_mu, contact_resistance=contact_resistance)

        # Cloth-specific properties
        self._thickness = thickness
        self._bending_stiffness = bending_stiffness
        self._model = model

    @property
    def thickness(self):
        """Shell thickness (m)."""
        return self._thickness

    @property
    def bending_stiffness(self):
        """Bending stiffness coefficient."""
        return self._bending_stiffness

    @property
    def model(self):
        """FEM material model name (unused for cloth)."""
        return self._model

    def __repr__(self):
        return f"<gs.materials.FEM.Cloth(E={self.E}, nu={self.nu}, rho={self.rho}, thickness={self.thickness})>"
