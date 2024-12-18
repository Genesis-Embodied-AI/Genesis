import taichi as ti

from ..rigid_entity import RigidLink
from .avatar_geom import AvatarGeom, AvatarVisGeom


@ti.data_oriented
class AvatarLink(RigidLink):

    def _add_geom(
        self, mesh, init_pos, init_quat, type, friction, sol_params, center_init=None, needs_coup=False, data=None
    ):
        geom = AvatarGeom(
            link=self,
            idx=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            mesh=mesh,
            init_pos=init_pos,
            init_quat=init_quat,
            type=type,
            friction=friction,
            sol_params=sol_params,
            center_init=center_init,
            needs_coup=needs_coup,
            data=data,
        )
        self._geoms.append(geom)

    def _add_vgeom(self, vmesh, init_pos, init_quat, type, data=None, surface=None):
        vgeom = AvatarVisGeom(
            link=self,
            idx=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            vmesh=vmesh,
            init_pos=init_pos,
            init_quat=init_quat,
            type=type,
            surface=surface,
        )
        self._vgeoms.append(vgeom)
