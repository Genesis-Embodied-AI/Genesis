
from dataclasses import dataclass
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import numpy as np
from typing import Dict, List, Optional
from scipy.spatial.transform import Rotation

def get_prim_type_name(prim: Usd.Prim):
    return prim.GetPrimTypeInfo().GetTypeName()

def gf_quat_to_rotation(gf_quat: Gf.Quatf) -> Rotation:
    real_part: float = gf_quat.GetReal()
    imag_part: Gf.Vec3f = gf_quat.GetImaginary()
    return Rotation.from_quat([imag_part[0], imag_part[1], imag_part[2], real_part])

@dataclass(frozen=True)
class PhysicsJoint:
    """Represents a physics joint in the USD stage.

    A physics joint connects two rigid bodies and constrains their relative movement.
    """

    # The prim of this joint in usd stage
    prim: Usd.Prim

    # The first link prim of this joint in usd stage
    body0: Optional[Usd.Prim]

    # The second link prim of this joint in usd stage
    body1: Optional[Usd.Prim]

    # Relative position of the joint frame to body0's frame.
    local_pos0: Optional[np.ndarray]

    # Relative orientation of the joint frame to body0's frame.
    local_orient0: Optional[Rotation]

    # Relative position of the joint frame to body1's frame.
    local_pos1: Optional[np.ndarray]

    # Relative orientation of the joint frame to body1's frame.
    local_orient1: Optional[Rotation]

@dataclass(frozen=True)
class RevoluteJoint(PhysicsJoint):
    """Represents a revolute joint in the USD stage.

    A revolute joint allows rotation around a single axis between two rigid bodies.
    It inherits basic joint properties from PhysicsJoint.
    """

    # The axis which this joint rotates about, root joint has no axis
    axis: Optional[str]

    # The lower limit of this joint in degrees
    lower_limit: Optional[float]

    # The upper limit of this joint in degrees
    upper_limit: Optional[float]


@dataclass(frozen=True)
class FixedJoint(PhysicsJoint):
    pass

class ArticulationBuilder:
    """Scene builder for articulation bodies."""

    def __init__(self):
        self.joint_velocity = {}  # computed velocity
        self.joint_position = {}  # position read from geo attr
        self.joint_instruct_velocity = {}
        self.joint_instruct_position = {}
        self.joint_names = []
        self.joint_geometry = {}
        self.joint_lower_limits = {}
        self.joint_upper_limits = {}
        self.metersPerUnit = 1.0 # Assume meters for simplicity

        self._articulation_bodies: Dict[str, Dict[str, List[PhysicsJoint]]] = {}

    def reset(self):
        self.joint_instruct_velocity.clear()
        self.joint_instruct_position.clear()

    def parse_usd(self, stage: Usd.Stage) -> None:
        """Extract articulation bodies from the USD stage.

        Args:
            stage: The USD stage to parse
        """

        # Get all articulation root prim path
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim_type: str = get_prim_type_name(prim)
                if prim_type == "Xform":
                    root_str = str(prim.GetPath())
                else:
                    root_str = str(prim.GetParent().GetPath())
                if root_str not in self._articulation_bodies:
                    self._articulation_bodies[root_str] = {}

        for prim_path in self._articulation_bodies.keys():
            root_prim = stage.GetPrimAtPath(prim_path)
            for child_prim in Usd.PrimRange(root_prim):
                if get_prim_type_name(child_prim) == "PhysicsJoint":
                    joint_api = UsdPhysics.Joint(child_prim)
                    joint = PhysicsJoint(
                        prim=child_prim,
                        body0=stage.GetPrimAtPath(
                            joint_api.GetBody0Rel().GetTargets()[0]
                        ),
                        body1=stage.GetPrimAtPath(
                            joint_api.GetBody1Rel().GetTargets()[0]
                        ),
                        local_pos0=np.array(joint_api.GetLocalPos0Attr().Get())
                        * self.metersPerUnit,
                        local_orient0=gf_quat_to_rotation(
                            joint_api.GetLocalRot0Attr().Get()
                        ),
                        local_pos1=np.array(joint_api.GetLocalPos1Attr().Get())
                        * self.metersPerUnit,
                        local_orient1=gf_quat_to_rotation(
                            joint_api.GetLocalRot1Attr().Get()
                        ),
                    )
                    self._articulation_bodies[prim_path]["physics_joints"] = [joint]
                elif get_prim_type_name(child_prim) == "PhysicsRevoluteJoint":
                    if (
                        self._articulation_bodies[prim_path].get("revolute_joints")
                        is None
                    ):
                        self._articulation_bodies[prim_path]["revolute_joints"] = []

                    joint_api = UsdPhysics.RevoluteJoint(child_prim)
                    joint = RevoluteJoint(
                        prim=child_prim,
                        body0=stage.GetPrimAtPath(
                            joint_api.GetBody0Rel().GetTargets()[0]
                        ),
                        body1=stage.GetPrimAtPath(
                            joint_api.GetBody1Rel().GetTargets()[0]
                        ),
                        axis=str(joint_api.GetAxisAttr().Get()),
                        lower_limit=float(joint_api.GetLowerLimitAttr().Get()),
                        upper_limit=float(joint_api.GetUpperLimitAttr().Get()),
                        local_pos0=np.array(joint_api.GetLocalPos0Attr().Get())
                        * self.metersPerUnit,
                        local_orient0=gf_quat_to_rotation(
                            joint_api.GetLocalRot0Attr().Get()
                        ),
                        local_pos1=np.array(joint_api.GetLocalPos1Attr().Get())
                        * self.metersPerUnit,
                        local_orient1=gf_quat_to_rotation(
                            joint_api.GetLocalRot1Attr().Get()
                        ),
                    )
                    self._articulation_bodies[prim_path]["revolute_joints"].append(
                        joint
                    )
                elif get_prim_type_name(child_prim) == "PhysicsFixedJoint":
                    if self._articulation_bodies[prim_path].get("fixed_joints") is None:
                        self._articulation_bodies[prim_path]["fixed_joints"] = []
                    joint_api = UsdPhysics.FixedJoint(child_prim)
                    body0_targets = joint_api.GetBody0Rel().GetTargets()
                    body1_targets = joint_api.GetBody1Rel().GetTargets()

                    joint = FixedJoint(
                        prim=child_prim,
                        body0=stage.GetPrimAtPath(body0_targets[0])
                        if body0_targets
                        else None,
                        body1=stage.GetPrimAtPath(body1_targets[0])
                        if body1_targets
                        else None,
                        local_pos0=np.array(joint_api.GetLocalPos0Attr().Get())
                        * self.metersPerUnit,
                        local_orient0=gf_quat_to_rotation(
                            joint_api.GetLocalRot0Attr().Get()
                        ),
                        local_pos1=np.array(joint_api.GetLocalPos1Attr().Get())
                        * self.metersPerUnit,
                        local_orient1=gf_quat_to_rotation(
                            joint_api.GetLocalRot1Attr().Get()
                        ),
                    )
                    self._articulation_bodies[prim_path]["fixed_joints"].append(joint)
                else:
                    pass

if __name__ == "__main__":
    stage = Usd.Stage.Open("D:\\MyStorage\\Project\\GenesisProject\\Genesis\\playground\\assets\\input_mesh.usda")
    builder = ArticulationBuilder()
    builder.parse_usd(stage)
    # print infos
    for art_root, joints_dict in builder._articulation_bodies.items():
        print(f"Articulation Root: {art_root}")
        for joint_type, joints in joints_dict.items():
            print(f"  Joint Type: {joint_type}")
            for joint in joints:
                print(f"    Joint Prim: {joint.prim.GetPath()}")
                print(f"      Body0: {joint.body0.GetPath() if joint.body0 else 'None'}")
                print(f"      Body1: {joint.body1.GetPath() if joint.body1 else 'None'}")
                if isinstance(joint, RevoluteJoint):
                    print(f"      Axis: {joint.axis}")
                    print(f"      Lower Limit: {joint.lower_limit}")
                    print(f"      Upper Limit: {joint.upper_limit}")