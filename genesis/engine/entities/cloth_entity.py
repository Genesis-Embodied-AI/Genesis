"""
Cloth Entity for IPC-based cloth simulation.

ClothEntity represents a thin shell/membrane object simulated using IPC.
It does not use the Genesis FEM solver and is managed entirely by IPCCoupler.
"""

import numpy as np
import torch
import gstaichi as ti

import genesis as gs
from .base_entity import Entity


@ti.data_oriented
class ClothEntity(Entity):
    """
    A cloth entity for thin shell simulation using IPC.

    This entity represents flexible cloth/fabric materials. Unlike FEMEntity,
    it only stores surface mesh (triangles) and is simulated exclusively through
    the IPC backend using shell-based FEM.

    Parameters
    ----------
    scene : Scene
        The simulation scene.
    material : Cloth
        Cloth material properties.
    morph : Morph
        Shape specification (must provide surface mesh).
    surface : Surface
        Rendering surface.
    idx : int
        Entity index in the scene.
    """

    def __init__(self, scene, material, morph, surface, idx):
        # ClothEntity doesn't have a dedicated solver, managed by IPCCoupler
        super().__init__(idx, scene, morph, solver=None, material=material, surface=surface)

        self._surface.update_texture()

        # Load and process mesh
        self._load_mesh()

        # Initialize state tracking
        self._positions = None  # Will be updated by IPC
        self._velocities = None

    def _load_mesh(self):
        """Load surface mesh from morph."""
        if isinstance(self.morph, gs.options.morphs.Mesh):
            # Use uipc's SimplicialComplexIO to load mesh, matching libuipc workflow
            from uipc.geometry import SimplicialComplexIO
            from uipc import Transform

            # Create transform for scaling, rotation, and translation
            # Order: scale -> rotate -> translate (standard transform order)
            transform = Transform.Identity()

            # 1. Apply scale first
            transform.scale(self._morph.scale)

            # 2. Apply rotation if specified (euler or quat)
            if self._morph.quat is not None:
                # Convert quaternion to rotation matrix, then apply
                from scipy.spatial.transform import Rotation as R
                quat_xyzw = [self._morph.quat[1], self._morph.quat[2], self._morph.quat[3], self._morph.quat[0]]  # w,x,y,z -> x,y,z,w
                rot = R.from_quat(quat_xyzw)
                euler_xyz = rot.as_euler('xyz', degrees=False)  # Get euler angles in radians

                # Apply rotation using AngleAxis (axis-angle representation)
                from uipc import AngleAxis, Vector3
                # Convert euler XYZ to sequential rotations
                if abs(euler_xyz[0]) > 1e-6:  # X rotation
                    transform.rotate(AngleAxis(euler_xyz[0], Vector3.UnitX()))
                if abs(euler_xyz[1]) > 1e-6:  # Y rotation
                    transform.rotate(AngleAxis(euler_xyz[1], Vector3.UnitY()))
                if abs(euler_xyz[2]) > 1e-6:  # Z rotation
                    transform.rotate(AngleAxis(euler_xyz[2], Vector3.UnitZ()))

            # 3. Apply translation last
            transform.translate(np.array(self._morph.pos))

            # Load mesh using uipc's loader
            io = SimplicialComplexIO(transform)
            self._uipc_base_mesh = io.read(self._morph.file)  # Store the uipc mesh

            # Extract vertices and faces for convenience
            verts = self._uipc_base_mesh.positions().view()
            faces = self._uipc_base_mesh.triangles().topo().view()

        else:
            gs.raise_exception(
                f"ClothEntity currently only supports Mesh morph. Got: {type(self.morph).__name__}"
            )

        # Store mesh data
        self.init_vertices = verts.astype(gs.np_float)
        self.faces = faces.astype(gs.np_int)
        self.n_vertices = len(self.init_vertices)
        self.n_faces = len(self.faces)

        gs.logger.info(
            f"ClothEntity mesh loaded: {self.n_vertices} vertices, {self.n_faces} faces"
        )

        # Rendering fields will be initialized later in build phase
        self._rendering_initialized = False

    def _init_rendering_fields(self, n_envs=1):
        """Initialize Taichi fields for rendering."""
        if self._rendering_initialized:
            return

        # Store n_envs for kernel use (like FEM solver's _B)
        self._n_envs = n_envs

        # Convert faces to Taichi field for kernel access
        self.faces_ti = ti.Vector.field(3, dtype=ti.i32, shape=(self.n_faces,))
        self.faces_ti.from_numpy(self.faces)

        # Rendering state structs (similar to FEM solver)
        surface_state_render_v = ti.types.struct(
            vertices=ti.types.vector(3, gs.ti_float),
        )
        surface_state_render_f = ti.types.struct(
            indices=ti.i32,
        )

        # Create rendering fields
        self.surface_render_v = surface_state_render_v.field(
            shape=(self.n_vertices, n_envs),
            layout=ti.Layout.SOA
        )
        self.surface_render_f = surface_state_render_f.field(
            shape=(self.n_faces * 3,),
            layout=ti.Layout.SOA
        )

        self._rendering_initialized = True

    def set_pos(self, frame, positions):
        """
        Update cloth vertex positions (called by IPCCoupler).

        Parameters
        ----------
        frame : int
            Frame index.
        positions : ndarray
            New vertex positions, shape (n_envs, n_vertices, 3).
        """
        self._positions = positions

        # Initialize rendering fields if not done yet
        if not self._rendering_initialized:
            n_envs = positions.shape[0] if len(positions.shape) > 1 else 1
            self._init_rendering_fields(n_envs)

        # Update rendering state
        self.get_state_render(frame)

    def get_pos(self):
        """Get current vertex positions."""
        if self._positions is None:
            # Return initial positions if not yet simulated
            return self.init_vertices
        return self._positions

    def get_state_render(self, f):
        """
        Update rendering state from current positions.

        Parameters
        ----------
        f : int
            Frame index.
        """
        if self._positions is None:
            # Use initial positions if not yet simulated
            pos_tensor = torch.from_numpy(self.init_vertices).to(gs.device)
            pos_tensor = pos_tensor.unsqueeze(0)  # Add batch dimension
        else:
            # Convert numpy positions to tensor
            pos_tensor = torch.from_numpy(self._positions).to(gs.device)

        # Update rendering vertices using kernel
        self._update_render_vertices_kernel(pos_tensor)

    @ti.kernel
    def _update_render_vertices_kernel(self, positions: ti.types.ndarray()):
        """
        Update rendering vertices and face indices from positions array.
        Similar to FEM solver's get_state_render_kernel.

        Parameters
        ----------
        positions : ndarray
            Shape (n_envs, n_vertices, 3)
        """
        # Update vertices (use stored _n_envs like FEM uses _B)
        for i_v, i_b in ti.ndrange(self.n_vertices, self._n_envs):
            for j in ti.static(range(3)):
                self.surface_render_v[i_v, i_b].vertices[j] = positions[i_b, i_v, j]

        # Update face indices (like FEM solver does every frame)
        for i_f in range(self.n_faces):
            for j in ti.static(range(3)):
                self.surface_render_f[i_f * 3 + j].indices = ti.cast(self.faces_ti[i_f][j], ti.i32)

    @property
    def is_built(self):
        """Check if entity is built."""
        return self._scene._is_built

    def __repr__(self):
        return f"<gs.ClothEntity uid={self.uid}, vertices={self.n_vertices}, faces={self.n_faces}>"
