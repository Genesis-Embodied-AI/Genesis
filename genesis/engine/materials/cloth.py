"""
Cloth material for IPC-based cloth simulation.

This material is used exclusively with IPCCoupler and ClothEntity.
"""

import genesis as gs
from .base import Material


class Cloth(Material):
    """
    Cloth material for thin shell/membrane simulation using IPC.

    This material is designed for cloth, fabric, and other thin flexible materials.
    It uses shell-based FEM formulation (NeoHookeanShell) in the IPC backend.

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

    Notes
    -----
    - Only works with IPCCoupler enabled
    - Requires GPU backend
    - Cannot be used with Genesis FEM solver
    - Only accepts surface mesh morphs (Mesh, etc.)

    Examples
    --------
    >>> cloth = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="cloth.obj"),
    ...     material=gs.materials.Cloth(
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
    ):
        super().__init__()

        self._E = E
        self._nu = nu
        self._rho = rho
        self._thickness = thickness
        self._bending_stiffness = bending_stiffness

    @property
    def E(self):
        """Young's modulus (Pa)."""
        return self._E

    @property
    def nu(self):
        """Poisson's ratio."""
        return self._nu

    @property
    def rho(self):
        """Material density (kg/m³)."""
        return self._rho

    @property
    def thickness(self):
        """Shell thickness (m)."""
        return self._thickness

    @property
    def bending_stiffness(self):
        """Bending stiffness coefficient."""
        return self._bending_stiffness

    def __repr__(self):
        return f"<gs.Cloth(E={self.E}, nu={self.nu}, rho={self.rho}, thickness={self.thickness})>"
