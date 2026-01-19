import re

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade

import genesis as gs
from genesis.utils import geom as gu

from .usd_context import UsdContext
from .usd_parser_utils import AXES_T



def geom_exception(geom_type, geom_id, stage_file, reason_msg):
    gs.raise_exception(f"{reason_msg} for {geom_type} {geom_id} in usd file {stage_file}.")


def parse_prim_geoms(
    context: UsdContext,
    prim: Usd.Prim,
    link_prim: Usd.Prim,
    morph: gs.morphs.USD,
    surface: gs.surfaces.Surface,
    match_visual=False,
    match_collision=False,
):
    g_infos = []

    if not match_visual:
        for pattern in morph.visual_mesh_prim_patterns:
            if re.match(pattern, prim.GetName()):
                match_visual = True
                break
    if not match_collision:
        for pattern in morph.collision_mesh_prim_patterns:
            if re.match(pattern, prim.GetName()):
                match_collision = True
                break

    if prim.IsA(UsdGeom.Gprim):
        # parse materials
        prim_bindings = UsdShade.MaterialBindingAPI(prim)
        material_prim = prim_bindings.ComputeBoundMaterial()[0]
        if material_prim.GetPrim().IsValid():
            surface_id = context.get_prim_id(material_prim.GetPrim())
            geom_surface, uv_name = context.apply_surface(surface_id, surface)
        else:
            geom_surface, uv_name, surface_id = surface.copy(), "st", None

        # parse transform
        geom_Q, geom_S = context.compute_gs_transform(prim, link_prim)
        geom_S *= morph.scale
        geom_Q[:3, 3] *= morph.scale
        geom_id = context.get_prim_id(prim)

        # parse geometry
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = UsdGeom.Mesh(prim)

            # parse vertices
            points_attr = mesh_prim.GetPointsAttr()
            if not points_attr.HasValue():
                geom_exception("Mesh", geom_id, morph.file, "No vertices")
            points = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)

            # parse faces
            faces_attr = mesh_prim.GetFaceVertexCountsAttr()
            faces_vertex_counts_attr = mesh_prim.GetFaceVertexIndicesAttr()
            faces = (
                np.array(faces_attr.Get(), dtype=np.int32) if faces_attr.HasValue() else np.array([], dtype=np.int32)
            )
            faces_vertex_counts = (
                np.array(faces_vertex_counts_attr.Get())
                if faces_vertex_counts_attr.HasValue()
                else np.array([], dtype=np.int32)
            )
            points_faces_varying = False

            # parse normals
            normals = None
            normals_attr = mesh_prim.GetNormalsAttr()
            if normals_attr.HasValue():
                normals = np.array(normals_attr.Get(), dtype=np.float32)
                if normals.shape[0] != points.shape[0]:
                    if normals.shape[0] == faces.shape[0]:  # face varying meshes, adjacent faces do not share vertices
                        points_faces_varying = True
                    else:
                        geom_exception("Mesh", geom_id, morph.file, "Size of normals mismatch")

            # parse UVs
            uvs = None
            if uv_name is not None:
                uv_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
                if uv_var.IsDefined() and uv_var.HasValue():
                    uvs = np.array(uv_var.ComputeFlattened(), dtype=np.float32)
                    uvs[:, 1] = 1.0 - uvs[:, 1]  # Flip V coordinate
                    if uvs.shape[0] != points.shape[0]:
                        if uvs.shape[0] == faces.shape[0]:
                            points_faces_varying = True
                        elif uvs.shape[0] == 1:
                            uvs = None
                        else:
                            geom_exception("Mesh", geom_id, morph.file, "Size of uvs mismatch")

            # process faces
            if faces_vertex_counts.size == 0:
                triangles = np.array([], dtype=np.int32).reshape(0, 3)
            else:
                # rearrange points and faces
                if points_faces_varying:
                    points = points[faces]
                    faces = np.arange(faces.shape[0])

                # triangulate faces
                if np.max(faces_vertex_counts) > 3:
                    triangles = []
                    bi = 0
                    for face_vertex_count in faces_vertex_counts:
                        if face_vertex_count == 3:
                            triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                        elif face_vertex_count > 3:
                            for i in range(1, face_vertex_count - 1):
                                triangles.append([faces[bi + 0], faces[bi + i], faces[bi + i + 1]])
                        bi += face_vertex_count
                    triangles = np.array(triangles, dtype=np.int32)
                else:
                    triangles = faces.reshape(-1, 3)

            # process mesh
            processed_mesh = trimesh.Trimesh(
                vertices=points,
                faces=triangles,
                vertex_normals=normals,
                visual=trimesh.visual.TextureVisuals(uv=uvs) if uvs is not None else None,
                process=True,
            )
            processed_mesh.apply_transform(geom_S)
            points = processed_mesh.vertices
            triangles = processed_mesh.faces
            normals = processed_mesh.vertex_normals
            if uvs is not None:
                uvs = processed_mesh.visual.uv

            metadata = {
                "mesh_path": context.stage_file,
            }  # unbaked file or cache
            mesh = gs.Mesh.from_attrs(
                verts=points,
                faces=triangles,
                normals=normals,
                surface=geom_surface,
                uvs=uvs,
            )
            mesh.metadata.update(metadata)
            geom_data = None
            gs_type = gs.GEOM_TYPE.MESH

        else:  # primitive geometries
            geom_S_diag = np.diag(geom_S)
            if prim.IsA(UsdGeom.Plane):
                plane_prim = UsdGeom.Plane(prim)
                width = plane_prim.GetWidthAttr().Get()
                length = plane_prim.GetLengthAttr().Get()
                axis_T = AXES_T[plane_prim.GetAxisAttr().Get() or "Z"]

                w = float(width) * 0.5
                l = float(length) * 0.5
                tmesh = trimesh.Trimesh(
                    vertices=np.array([[-w, -l, 0.0], [w, -l, 0.0], [w, l, 0.0], [-w, l, 0.0]], dtype=np.float32),
                    faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                    face_normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                )
                tmesh.apply_transform(axis_T)
                geom_data = np.array([0.0, 0.0, 1.0])
                gs_type = gs.GEOM_TYPE.PLANE

            elif prim.IsA(UsdGeom.Sphere):
                sphere_prim = UsdGeom.Sphere(prim)
                radius = sphere_prim.GetRadiusAttr().Get()
                tmesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
                geom_data = np.array([radius]) * geom_S_diag
                gs_type = gs.GEOM_TYPE.SPHERE

            elif prim.IsA(UsdGeom.Capsule):
                capsule_prim = UsdGeom.Capsule(prim)
                radius = capsule_prim.GetRadiusAttr().Get()
                height = capsule_prim.GetHeightAttr().Get()
                axis_T = AXES_T[capsule_prim.GetAxisAttr().Get() or "Z"]
                tmesh = trimesh.creation.capsule(radius=radius, height=height, count=(8, 12))
                tmesh.apply_translation([0.0, 0.0, -0.5 * height])
                tmesh.apply_transform(axis_T)
                geom_data = np.array([radius, height, 1.0]) * geom_S_diag  # TODO: use the correct direction
                gs_type = gs.GEOM_TYPE.CAPSULE

            elif prim.IsA(UsdGeom.Cube):
                cube_prim = UsdGeom.Cube(prim)
                size = cube_prim.GetSizeAttr().Get()
                extents = np.array([size, size, size], dtype=np.float32)
                tmesh = trimesh.creation.box(extents=extents)
                geom_data = extents * geom_S_diag
                gs_type = gs.GEOM_TYPE.BOX

            elif prim.IsA(UsdGeom.Cylinder):
                cylinder_prim = UsdGeom.Cylinder(prim)
                radius = cylinder_prim.GetRadiusAttr().Get()
                height = cylinder_prim.GetHeightAttr().Get()
                axis_T = AXES_T[cylinder_prim.GetAxisAttr().Get() or "Z"]
                tmesh = trimesh.creation.cylinder(radius=radius, height=height, count=(8, 12))
                tmesh.apply_transform(axis_T)
                geom_data = np.array([radius, height, 1.0]) * geom_S_diag  # TODO: use the correct direction
                gs_type = gs.GEOM_TYPE.CYLINDER

            else:
                gs.raise_exception(f"Unsupported geometry type: {prim.GetTypeName()}")

            tmesh.apply_transform(geom_S)
            mesh = gs.Mesh.from_trimesh(
                tmesh,
                surface=geom_surface,
            )

        geom_pos = geom_Q[:3, 3]
        geom_quat = gu.R_to_quat(geom_Q[:3, :3])

        is_visual = match_visual or not (match_collision or match_visual)
        is_collision = match_collision or not (match_collision or match_visual)
        if is_visual:
            g_infos.append(
                dict(
                    vmesh=mesh,
                    pos=geom_pos,
                    quat=geom_quat,
                    contype=0,
                    conaffinity=0,
                    type=gs_type,
                    data=geom_data,
                )
            )
        if is_collision:
            # TODO: use "physics:material:binding" to extract frictions
            g_infos.append(
                dict(
                    mesh=mesh,
                    pos=geom_pos,
                    quat=geom_quat,
                    contype=1,
                    conaffinity=1,
                    type=gs_type,
                    data=geom_data,
                    friction=gu.default_friction(),
                    sol_params=gu.default_solver_params(),
                )
            )

    for child in prim.GetChildren():
        g_infos.extend(parse_prim_geoms(context, child, link_prim, morph, surface, match_visual, match_collision))

    return g_infos
