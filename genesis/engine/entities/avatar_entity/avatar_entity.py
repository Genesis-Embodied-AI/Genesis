import taichi as ti

from ..rigid_entity import RigidEntity
from .avatar_joint import AvatarJoint
from .avatar_link import AvatarLink


@ti.data_oriented
class AvatarEntity(RigidEntity):
    def add_link(
        self,
        name,
        pos,
        quat,
        inertial_pos,
        inertial_quat,
        inertial_i,
        inertial_mass,
        parent_idx,
        invweight,
    ):
        link = AvatarLink(
            entity=self,
            name=name,
            idx=self.n_links + self._link_start,
            geom_start=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            pos=pos,
            quat=quat,
            inertial_pos=inertial_pos,
            inertial_quat=inertial_quat,
            inertial_i=inertial_i,
            inertial_mass=inertial_mass,
            parent_idx=parent_idx,
            invweight=invweight,
            visualize_contact=False,
        )
        self._links.append(link)
        return link

    def add_joint(
        self,
        name,
        n_qs,
        n_dofs,
        type,
        pos,
        quat,
        dofs_motion_ang,
        dofs_motion_vel,
        dofs_limit,
        dofs_invweight,
        dofs_stiffness,
        dofs_sol_params,
        dofs_damping,
        dofs_armature,
        dofs_kp,
        dofs_kv,
        dofs_force_range,
        init_q,
    ):
        joint = AvatarJoint(
            entity=self,
            name=name,
            idx=self.n_joints + self._joint_start,
            q_start=self.n_qs + self._q_start,
            dof_start=self.n_dofs + self._dof_start,
            n_qs=n_qs,
            n_dofs=n_dofs,
            type=type,
            pos=pos,
            quat=quat,
            dofs_motion_ang=dofs_motion_ang,
            dofs_motion_vel=dofs_motion_vel,
            dofs_limit=dofs_limit,
            dofs_invweight=dofs_invweight,
            dofs_stiffness=dofs_stiffness,
            dofs_sol_params=dofs_sol_params,
            dofs_damping=dofs_damping,
            dofs_armature=dofs_armature,
            dofs_kp=dofs_kp,
            dofs_kv=dofs_kv,
            dofs_force_range=dofs_force_range,
            init_q=init_q,
        )
        self._joints.append(joint)
        return joint

    def init_jac_and_IK(self):
        # TODO: Avatar should also support IK
        pass
