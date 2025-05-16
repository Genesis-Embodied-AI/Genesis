import numpy as np
import numpy.typing as npt

import taichi as ti

from ..rigid_entity import RigidEntity
from .avatar_joint import AvatarJoint
from .avatar_link import AvatarLink


@ti.data_oriented
class AvatarEntity(RigidEntity):
    def add_link(
        self,
        name: str,
        pos,
        quat,
        inertial_pos,
        inertial_quat,
        inertial_i,
        inertial_mass: float,
        parent_idx: int,
        invweight: npt.NDArray[np.float64],
    ) -> AvatarLink:
        """
        Add a new link (AvatarLink) to the entity.

        Parameters
        ----------
        name : str
            Name of the link.
        pos : array-like
            Position of the link in world or parent frame.
        quat : array-like
            Orientation (quaternion) of the link.
        inertial_pos : array-like
            Position of the inertial frame relative to the link.
        inertial_quat : array-like
            Orientation of the inertial frame.
        inertial_i : array-like
            Inertia tensor in the local frame.
        inertial_mass : float
            Mass of the link.
        parent_idx : int
            Index of the parent link in the kinematic tree.
        invweight : np array of 2 float elements
            Inverse weight for optimization or simulation purposes.

        Returns
        -------
        link : AvatarLink
            The created AvatarLink instance.
        """
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
        name: str,
        n_qs: int,
        n_dofs: int,
        type: str,
        pos,
        quat,
        dofs_motion_ang,
        dofs_motion_vel,
        dofs_limit,
        dofs_invweight,
        dofs_stiffness,
        dofs_damping,
        dofs_armature,
        dofs_kp,
        dofs_kv,
        dofs_force_range,
        init_q,
    ) -> AvatarJoint:
        """
        Add a new joint (AvatarJoint) to the entity.

        Parameters
        ----------
        name : str
            Name of the joint.
        n_qs : int
            Number of configuration variables (generalized coordinates).
        n_dofs : int
            Number of degrees of freedom for the joint.
        type : str
            Type of the joint (e.g., "revolute", "prismatic").
        pos : array-like
            Position of the joint frame.
        quat : array-like
            Orientation (quaternion) of the joint frame.
        dofs_motion_ang : array-like
            Angular motions allowed for each DOF.
        dofs_motion_vel : array-like
            Velocity directions for each DOF.
        dofs_limit : array-like
            Limits for each DOF (e.g., min/max).
        dofs_invweight : array-like
            Inverse weight for each DOF.
        dofs_stiffness : array-like
            Stiffness values for each DOF.
        dofs_damping : array-like
            Damping values for each DOF.
        dofs_armature : array-like
            Armature inertia values.
        dofs_kp : array-like
            Proportional gains for control.
        dofs_kv : array-like
            Derivative gains for control.
        dofs_force_range : array-like
            Allowed force/torque range for each DOF.
        init_q : array-like
            Initial configuration (position/orientation) for the joint.

        Returns
        -------
        joint : AvatarJoint
            The created AvatarJoint instance.
        """
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
            dofs_damping=dofs_damping,
            dofs_armature=dofs_armature,
            dofs_kp=dofs_kp,
            dofs_kv=dofs_kv,
            dofs_force_range=dofs_force_range,
            init_q=init_q,
        )
        self._joints.append(joint)
        return joint

    def init_jac_and_IK(self) -> None:
        # TODO: Avatar should also support IK
        pass
