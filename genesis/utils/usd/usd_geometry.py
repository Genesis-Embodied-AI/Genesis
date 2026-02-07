import re
from typing import Dict, List

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

import genesis as gs
from genesis.utils import geom as gu

from .usd_context import UsdContext
from .usd_utils import AXES_T, usd_attr_array_to_numpy, usd_primvar_array_to_numpy


def geom_exception(geom_type, geom_id, stage_file, reason_msg):
    gs.raise_exception(f"{reason_msg} for {geom_type} {geom_id} in usd file {stage_file}.")


def get_triangle_ids(tri_starts, tri_counts):
    tri_bases = np.repeat(tri_starts, tri_counts)
    tri_offsets = np.arange(tri_counts.sum(), dtype=np.int32)
    tri_stages = np.repeat(np.cumsum(tri_counts, dtype=np.int32) - tri_counts, tri_counts)
    return tri_bases + tri_offsets - tri_stages


def parse_prim_geoms(
    context: UsdContext,
    prim: Usd.Prim,
    link_prim: Usd.Prim,
    links_g_infos: List[List[Dict]],
    link_path_to_idx: Dict[str, int],
    morph: gs.morphs.USD,
    surface: gs.surfaces.Surface,
    match_visual=False,
    match_collision=False,
):
    if not prim.IsActive():
        return

    if str(prim.GetPath()) in link_path_to_idx:
        link_prim = prim

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

    if link_prim is not None and prim.IsA(UsdGeom.Gprim):
        # parse materials
        geom_surface, geom_uvname, _surface_id, bake_success = context.apply_surface(prim, surface)
        gprim = UsdGeom.Gprim(prim)
        uvs = {geom_uvname: None}

        # parse transform
        geom_Q, geom_S = context.compute_gs_transform(prim, link_prim)
        geom_S *= morph.scale
        geom_ST = np.eye(4, dtype=geom_S.dtype)
        geom_ST[:3, :3] = geom_S
        geom_Q[:3, 3] *= morph.scale
        geom_id = context.get_prim_id(prim)

        # parse geometry
        meshes = []
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = UsdGeom.Mesh(prim)

            # parse vertices
            points = usd_attr_array_to_numpy(mesh_prim.GetPointsAttr(), np.float32)
            if points.size == 0:
                geom_exception("Mesh", geom_id, morph.file, "No vertices")

            # parse faces
            faces = usd_attr_array_to_numpy(mesh_prim.GetFaceVertexIndicesAttr(), np.int32)
            face_vertex_counts = usd_attr_array_to_numpy(mesh_prim.GetFaceVertexCountsAttr(), np.int32)
            points_faces_varying = False

            # parse normals
            normals = usd_attr_array_to_numpy(mesh_prim.GetNormalsAttr(), np.float32, True)
            if normals is not None and normals.shape[0] != points.shape[0]:
                if normals.shape[0] == faces.shape[0]:  # face varying meshes, adjacent faces do not share vertices
                    points_faces_varying = True
                else:
                    geom_exception("Mesh", geom_id, morph.file, "Size of normals mismatch")

            # parse geom subsets
            subset_infos = []
            face_used_mask = np.full(len(face_vertex_counts), False, dtype=np.bool_)
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh_prim)
            for subset in subsets:
                subset_prim = subset.GetPrim()
                elem_type = str(subset.GetElementTypeAttr().Get() or "face")
                if str(elem_type) == "face":
                    subset_face_ids_attr = subset.GetIndicesAttr()
                    subset_face_ids = usd_attr_array_to_numpy(subset_face_ids_attr, np.int32)
                    if subset_face_ids.size == 0:
                        continue
                    face_used_mask[subset_face_ids] = True
                    subset_surface, subset_uvname, _, subset_bake_success = context.apply_surface(subset_prim, surface)
                    subset_geom_id = context.get_prim_id(subset_prim)
                    subset_infos.append(
                        (subset_face_ids, subset_surface, subset_uvname, subset_geom_id, subset_bake_success)
                    )
                    uvs[subset_uvname] = None
                else:
                    gs.logger.warning(f"Unsupported geom subset element type: {elem_type} for {geom_id}")
            subset_unused = ~face_used_mask
            if subset_unused.any():
                subset_infos.append((subset_unused, geom_surface, geom_uvname, geom_id, bake_success))

            # parse UVs
            for uvname in uvs.keys():
                uv = usd_primvar_array_to_numpy(UsdGeom.PrimvarsAPI(prim).GetPrimvar(uvname), np.float32, True)
                if uv is not None:
                    uv[:, 1] = 1.0 - uv[:, 1]  # Flip V coordinate
                    if uv.shape[0] != points.shape[0]:
                        if uv.shape[0] == faces.shape[0]:
                            points_faces_varying = True
                        elif uv.shape[0] == 1:
                            uv = None
                        else:
                            geom_exception("Mesh", geom_id, morph.file, "Size of uvs mismatch")
                    uvs[uvname] = uv

            # process faces
            if face_vertex_counts.size == 0:
                triangles = np.empty((0, 3), dtype=np.int32)
                face_triangle_starts = np.empty(0, dtype=np.int32)
            else:
                # rearrange points and faces
                if points_faces_varying:
                    if normals is not None and normals.shape[0] == points.shape[0]:
                        normals = normals[faces]
                    for uvname in uvs.keys():
                        uv = uvs[uvname]
                        if uv is not None and uv.shape[0] == points.shape[0]:
                            uvs[uvname] = uv[faces]
                    points = points[faces]
                    faces = np.arange(faces.shape[0], dtype=np.int32)

                # triangulate faces
                # TODO: discard degenerated faces
                if np.max(face_vertex_counts) > 3:
                    triangles, face_triangle_starts = [], []
                    bi, ti = 0, 0
                    for face_vertex_count in face_vertex_counts:
                        face_triangle_starts.append(ti)
                        if face_vertex_count == 3:
                            triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                        elif face_vertex_count > 3:
                            for i in range(1, face_vertex_count - 1):
                                triangles.append([faces[bi + 0], faces[bi + i], faces[bi + i + 1]])
                        bi += face_vertex_count
                        ti += face_vertex_count - 2
                    triangles = np.asarray(triangles, dtype=np.int32)
                    face_triangle_starts = np.asarray(face_triangle_starts, dtype=np.int32)
                else:
                    triangles = faces.reshape(-1, 3)
                    face_triangle_starts = np.arange(len(face_vertex_counts), dtype=np.int32)

            # process mesh
            for subset_face_ids, subset_surface, subset_uvname, subset_geom_id, subset_bake_success in subset_infos:
                tri_starts = face_triangle_starts[subset_face_ids]
                tri_counts = face_vertex_counts[subset_face_ids] - 2
                tri_ids = get_triangle_ids(tri_starts, tri_counts)
                subset_triangles = triangles[tri_ids]
                subset_uv = uvs[subset_uvname]

                processed_mesh = trimesh.Trimesh(
                    vertices=points,
                    faces=subset_triangles,
                    vertex_normals=normals,
                    visual=trimesh.visual.TextureVisuals(uv=subset_uv) if subset_uv is not None else None,
                    process=True,
                )
                # TODO: use a more efficient custom function to remove unreferenced vertices
                processed_mesh.remove_unreferenced_vertices()
                processed_mesh.apply_transform(geom_ST)
                subset_points = processed_mesh.vertices
                subset_triangles = processed_mesh.faces
                subset_normals = processed_mesh.vertex_normals
                if subset_uv is not None:
                    subset_uv = processed_mesh.visual.uv

                mesh = gs.Mesh.from_attrs(
                    verts=subset_points,
                    faces=subset_triangles,
                    normals=subset_normals,
                    surface=subset_surface,
                    uvs=subset_uv,
                )
                mesh.metadata.update(
                    {
                        "mesh_path": context.stage_file,  # unbaked file or cache
                        "name": subset_geom_id,
                        "bake_success": subset_bake_success,
                    }
                )
                meshes.append(mesh)

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
                # TODO: create different trimesh for visual and collision
                tmesh = trimesh.creation.capsule(radius=radius, height=height, count=(8, 12))
                tmesh.apply_transform(axis_T)
                geom_data = np.array([radius, height, 1.0]) * geom_S_diag  # TODO: use the correct direction
                gs_type = gs.GEOM_TYPE.CAPSULE

            elif prim.IsA(UsdGeom.Cube):
                cube_prim = UsdGeom.Cube(prim)
                size = cube_prim.GetSizeAttr().Get()
                extents = np.array([size, size, size], dtype=np.float32)
                tmesh = trimesh.creation.box(extents=extents)
                geom_data = extents * geom_S_diag
                geom_surface.smooth = False
                gs_type = gs.GEOM_TYPE.BOX

            elif prim.IsA(UsdGeom.Cylinder):
                cylinder_prim = UsdGeom.Cylinder(prim)
                radius = cylinder_prim.GetRadiusAttr().Get()
                height = cylinder_prim.GetHeightAttr().Get()
                axis_T = AXES_T[cylinder_prim.GetAxisAttr().Get() or "Z"]
                tmesh = trimesh.creation.cylinder(radius=radius, height=height, count=(8, 12))
                tmesh.apply_transform(axis_T)
                geom_data = np.array([radius, height, 1.0]) * geom_S_diag  # TODO: use the correct direction
                geom_surface.smooth = False
                gs_type = gs.GEOM_TYPE.CYLINDER

            else:
                gs.raise_exception(f"Unsupported geometry type: {prim.GetTypeName()}")

            tmesh.apply_transform(geom_ST)
            metadata = {
                "name": geom_id,
                "bake_success": bake_success,
            }
            meshes.append(gs.Mesh.from_trimesh(tmesh, surface=geom_surface, metadata=metadata))

        geom_pos = geom_Q[:3, 3]
        geom_quat = gu.R_to_quat(geom_Q[:3, :3])

        is_guide = str(gprim.GetPurposeAttr().Get() or "default") == "guide"
        is_visible = str(gprim.ComputeVisibility()) != "invisible"
        is_visual = (is_visible and not is_guide) and (match_visual or not (match_collision or match_visual))
        is_collision = (is_visible) and (match_collision or not (match_collision or match_visual))

        g_infos = links_g_infos[link_path_to_idx[str(link_prim.GetPath())]]
        if is_visual:
            for mesh in meshes:
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
            # TODO: use "physics:material:binding" (UsdPhysicsMaterialAPI) to extract frictions
            for mesh in meshes:
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
        parse_prim_geoms(
            context, child, link_prim, links_g_infos, link_path_to_idx, morph, surface, match_visual, match_collision
        )
