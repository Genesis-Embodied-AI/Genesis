import numpy as np
import polyscope as ps
from polyscope import imgui
import mujoco

from uipc import view
from uipc import Scene, World, Engine, Transform, Vector12, Animation, Logger, Timer
from uipc import builtin
from uipc.unit import MPa, GPa
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, trimesh, label_surface, linemesh, affine_body
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyRevoluteJoint,
    AffineBodyPrismaticJoint,
    ExternalArticulationConstraint,
)
from uipc.gui import SceneGUI
from asset_dir import AssetDir
from mjcf import load_collision_robot


Logger.set_level(Logger.Level.Info)

this_output_path = AssetDir.output_path(__file__)
xml_path = f"{AssetDir.asset_path()}/xml/franka_emika_panda/panda_non_overlap.xml"
print(f"xml_path: {xml_path}")

engine = Engine("cuda", this_output_path)
world = World(engine)

dt = 0.01
config = Scene.default_config()
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = False
config["newton"]["velocity_tol"] = 0.1
config["newton"]["transrate_tol"] = 10
config["linear_system"]["tol_rate"] = 1e-4
config["contact"]["d_hat"] = 0.001
config["collision_detection"]["method"] = "stackless_bvh"
config["newton"]["semi_implicit"] = True
config["dt"] = dt

scene = Scene(config)

# Setup contact
scene.contact_tabular().default_model(0.05, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# Load MuJoCo model and collision meshes
model, data, collision_bodies, joints = load_collision_robot(xml_path)

# Create constitutions
abd = AffineBodyConstitution()

links = scene.objects().create("links")
body_slots = {}

def _make_transform_matrix(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rotation
    mat[:3, 3] = position
    return mat

for body in collision_bodies:
    mesh = trimesh(body.vertices, body.faces)
    # mesh = cube_mesh.copy()
    mesh.instances().resize(1)
    label_surface(mesh)
    
    io = SimplicialComplexIO()
    io.write(f"{this_output_path}/{body.name}.obj", mesh)
    
    print(f"body name: {body.name}")
    abd.apply_to(mesh, 100.0 * MPa)
    default_element.apply_to(mesh)

    transform_view = view(mesh.transforms())
    transform_view[0] = _make_transform_matrix(body.rotation, body.position)

    ref_dof_prev = mesh.instances().create("ref_dof_prev", Vector12.Zero())
    ref_dof_prev_view = view(ref_dof_prev)
    ref_dof_prev_view[:] = affine_body.transform_to_q(transform_view)

    external_kinetic = mesh.instances().find(builtin.external_kinetic)
    view(external_kinetic)[:] = 1

    is_fixed = mesh.instances().find(builtin.is_fixed)
    is_fixed_view = view(is_fixed)
    is_fixed_view[:] = 1 if model.body_parentid[body.body_id] == 0 else 0

    geo_slot, _ = links.geometries().create(mesh)
    body_slots[body.body_id] = (geo_slot, mesh)


def update_ref_dof_prev(info: Animation.UpdateInfo):
    for geo_slot in info.geo_slots():
        geo: SimplicialComplex = geo_slot.geometry()
        ref_dof_prev = geo.instances().find("ref_dof_prev")
        ref_dof_prev_view = view(ref_dof_prev)
        transform_view = view(geo.transforms())
        ref_dof_prev_view[:] = affine_body.transform_to_q(transform_view)


scene.animator().insert(links, update_ref_dof_prev)

# Create joints based on MuJoCo joints
joints_object = scene.objects().create("joints")
joint_geo_slots = []
joint_types = []
joint_names = []
delta_theta_tilde = np.zeros(0, dtype=np.float32)

axis_length = 0.1

for joint in joints:
    if joint.parent_body_id not in body_slots or joint.body_id not in body_slots:
        continue

    axis = joint.world_axis
    axis_norm = axis / (np.linalg.norm(axis) + 1e-8)
    p0 = joint.world_pos - axis_norm * axis_length
    p1 = joint.world_pos + axis_norm * axis_length

    Vs = np.array([p0, p1], dtype=np.float32)
    Es = np.array([[0, 1]], dtype=np.int32)
    joint_mesh = linemesh(Vs, Es)
    label_surface(joint_mesh)

    if joint.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        joint_constitution = AffineBodyRevoluteJoint()
    else:
        joint_constitution = AffineBodyPrismaticJoint()

    l_geo_slots = [body_slots[joint.parent_body_id][0]]
    l_instance_id = [0]
    r_geo_slots = [body_slots[joint.body_id][0]]
    r_instance_id = [0]
    strength_ratios = [100.0]

    joint_constitution.apply_to(
        joint_mesh,
        l_geo_slots,
        l_instance_id,
        r_geo_slots,
        r_instance_id,
        strength_ratios,
    )

    joint_slot, _ = joints_object.geometries().create(joint_mesh)
    joint_geo_slots.append(joint_slot)
    joint_types.append(joint.joint_type)
    joint_names.append(joint.name)


if joint_geo_slots:
    eac = ExternalArticulationConstraint()
    indices = [0] * len(joint_geo_slots)
    articulation = eac.create_geometry(joint_geo_slots, indices)

    joint_count = len(joint_geo_slots)
    mass_mat = np.eye(joint_count, dtype=np.float32)
    import test_mass
    mass_mat = test_mass.mass_mat
    mass = articulation["joint_joint"].find("mass")
    view(mass)[:] = mass_mat.flatten()

    articulation_object = scene.objects().create("articulation_object")
    articulation_object.geometries().create(articulation)

    delta_theta_tilde = np.zeros(joint_count, dtype=np.float32)

    def update_articulation(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        delta_attr = geo["joint"].find("delta_theta_tilde")
        delta_view = view(delta_attr)
        delta_view[:] = delta_theta_tilde * info.dt()

        mass_attr = geo["joint_joint"].find("mass")
        view(mass_attr)[:] = mass_mat.flatten()

    scene.animator().insert(articulation_object, update_articulation)


world.init(scene)
sgui = SceneGUI(scene, "split")

ps.init()
ps.set_ground_plane_height(-1.0)
sgui.register()
sgui.set_edge_width(1)
ps.set_up_dir("z_up")

run = False


def on_update():
    global run, delta_theta_tilde

    if imgui.Button("Run & Stop"):
        run = not run

    imgui.Separator()
    imgui.Text("External Articulation Control")
    imgui.Text("Adjust delta_theta_tilde values:")

    if joint_geo_slots:
        delta_theta_tilde[:] = 0.0
        for i, (name, jtype) in enumerate(zip(joint_names, joint_types)):
            if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                label = f"{name} (rad/s)"
                _, delta_theta_tilde[i] = imgui.SliderFloat(label, delta_theta_tilde[i], -5.0, 5.0)
            else:
                label = f"{name} (m/s)"
                _, delta_theta_tilde[i] = imgui.SliderFloat(label, delta_theta_tilde[i], -0.5, 0.5)

    # imgui.Separator()
    # imgui.Text(f"Frame: {world.frame()}")
    # imgui.Text(f"Time: {world.frame() * dt:.2f}s")

    if run:
        world.advance()
        world.retrieve()
        sgui.update()
        Timer.report()


ps.set_user_callback(on_update)
ps.show()
