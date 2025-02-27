import os
import pickle as pkl
from contextlib import redirect_stdout

import numpy as np
import pyvista as pv
import tetgen

import genesis as gs
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.ext import trimesh
from genesis.repr_base import RBC


class Mesh(RBC):
    """
    Genesis's own triangle mesh object.
    This is a wrapper of `trimesh.Trimesh` with some additional features and attributes. The internal trimesh object can be accessed via `self.trimesh`.
    We perform both convexification and decimation to preprocess the mesh for simulation if specified.

    Parameters
    ----------
    surface : genesis.Surface
        The mesh's surface object.
    uvs : np.ndarray
        The mesh's uv coordinates.
    convexify : bool
        Whether to convexify the mesh.
    decimate : bool
        Whether to decimate the mesh.
    decimate_face_num : int
        The target number of faces after decimation.
    metadata : dict
        The metadata of the mesh.
    """

    def __init__(
        self,
        mesh,
        surface=None,
        uvs=None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        metadata=dict(),
    ):
        self._uid = gs.UID()
        self._mesh = mesh
        self._surface = surface
        self._uvs = uvs
        self._metadata = metadata

        if self._surface.requires_uv():  # check uvs here
            if self._uvs is None:
                if "mesh_path" in metadata:
                    gs.logger.warning(
                        f"Texture given but asset missing uv info (or failed to load): {metadata['mesh_path']}"
                    )
                else:
                    gs.logger.warning("Texture given but asset missing uv info (or failed to load).")
        else:
            self._uvs = None

        if convexify:
            self.convexify()

        if decimate:
            self.decimate(decimate_face_num, convexify)

    def convexify(self):
        """
        Convexify the mesh.
        """
        if self._mesh.vertices.shape[0] > 3:
            self._mesh = trimesh.convex.convex_hull(self._mesh)
        self.clear_visuals()

    def decimate(self, target_face_num, convexify):
        """
        Decimate the mesh.
        """
        if self._mesh.vertices.shape[0] > 3 and self._mesh.faces.shape[0] > target_face_num:
            self._mesh = self._mesh.simplify_quadric_decimation(target_face_num)

            # need to run convexify again after decimation, because sometimes decimating a convex-mesh can make it non-convex...
            if convexify:
                self.convexify()

        self.clear_visuals()

    def remesh(self, edge_len_abs=None, edge_len_ratio=0.01, fix=True):
        """
        Remesh for tetrahedralization.
        """
        rm_file_path = mu.get_remesh_path(self.verts, self.faces, edge_len_abs, edge_len_ratio, fix)

        is_cached_loaded = False
        if os.path.exists(rm_file_path):
            gs.logger.debug("Remeshed file (`.rm`) found in cache.")
            try:
                with open(rm_file_path, "rb") as file:
                    verts, faces = pkl.load(file)
                is_cached_loaded = True
            except (EOFError, pkl.UnpicklingError):
                gs.logger.info("Ignoring corrupted cache.")

        if not is_cached_loaded:
            gs.logger.info("Remeshing for tetrahedralization...")
            with open(os.devnull, "w") as stdout, redirect_stdout(stdout):
                import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=self.verts, face_matrix=self.faces))
            if edge_len_abs is not None:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(edge_len_abs))
            else:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(edge_len_ratio * 100))
            m = ms.current_mesh()
            verts, faces = m.vertex_matrix(), m.face_matrix()
            # Maybe we need to fix the mesh in some extreme cases with open3d
            # if fix:
            #     verts, faces = pymeshfix.clean_from_arrays(verts, faces)
            os.makedirs(os.path.dirname(rm_file_path), exist_ok=True)
            with open(rm_file_path, "wb") as file:
                pkl.dump((verts, faces), file)

        self._mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
        )
        self.clear_visuals()

    def tetrahedralize(self, order, mindihedral, minratio, nobisect, quality, verbose):
        """
        Tetrahedralize the mesh.
        """
        pv_obj = pv.PolyData(
            self.verts, np.concatenate([np.full((self.faces.shape[0], 1), self.faces.shape[1]), self.faces], axis=1)
        )
        tet = tetgen.TetGen(pv_obj)
        verts, elems = tet.tetrahedralize(
            order=order, mindihedral=mindihedral, minratio=minratio, nobisect=nobisect, quality=quality, verbose=verbose
        )
        # visualize_tet(tet, pv_obj, show_surface=False, plot_cell_qual=False)
        return verts, elems

    def particlize(
        self,
        p_size=0.01,
        sampler="random",
    ):
        """
        Sample particles using the mesh volume.
        """
        if "pbs" in sampler:
            positions = pu.trimesh_to_particles_pbs(self._mesh, p_size, sampler)
            if positions is None:
                gs.logger.warning("`pbs` sampler failed. Falling back to `random` sampler.")
                sampler = "random"

        if sampler in ["random", "regular"]:
            positions = pu.trimesh_to_particles_simple(self._mesh, p_size, sampler)

        return positions

    def clear_visuals(self):
        """
        Clear the mesh's visual attributes by resetting the surface to gs.surfaces.Default().
        """
        self._surface = gs.surfaces.Default()
        self._surface.update_texture()

    def get_unique_edges(self):
        """
        Get the unique edges of the mesh.
        """
        r_face = np.roll(self.faces, 1, axis=1)
        edges = np.concatenate(np.array([self.faces, r_face]).T)

        # do a first pass to remove duplicates
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]

        return edges

    def copy(self):
        """
        Copy the mesh.
        """
        return Mesh(
            mesh=self._mesh.copy(),
            surface=self._surface.copy(),
            uvs=self._uvs.copy() if self._uvs is not None else None,
            metadata=self._metadata.copy(),
        )

    @classmethod
    def from_trimesh(
        cls, mesh, scale=None, convexify=False, decimate=False, decimate_face_num=500, metadata=dict(), surface=None
    ):
        """
        Create a genesis.Mesh from a trimesh.Trimesh object.
        """
        if surface is None:
            surface = gs.surfaces.Default()
        else:
            surface = surface.copy()
        mesh = mesh.copy()

        try:  # always parse uvs because roughness and normal map also need uvs
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]  # trimesh uses uvs starting from top left corner
        except:
            uvs = None

        roughness_factor = None
        color_image = None
        color_factor = None
        opacity = 1.0

        if mesh.visual.defined:
            if mesh.visual.kind == "texture":
                material = mesh.visual.material

                # TODO: Parsing PBR in obj or not
                # trimesh from .obj file will never use PBR material, but that from .glb file will
                if isinstance(material, trimesh.visual.material.PBRMaterial):
                    # color_image = None
                    # color_factor = None
                    if material.baseColorTexture is not None:
                        color_image = mu.PIL_to_array(material.baseColorTexture)
                    if material.baseColorFactor is not None:
                        color_factor = tuple(np.array(material.baseColorFactor, dtype=float) / 255.0)

                    if material.roughnessFactor is not None:
                        roughness_factor = (material.roughnessFactor,)

                elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                    if material.image is not None:
                        color_image = mu.PIL_to_array(material.image)
                    elif material.diffuse is not None:
                        color_factor = tuple(np.array(material.diffuse, dtype=float) / 255.0)

                    if material.glossiness is not None:
                        roughness_factor = ((2 / (material.glossiness + 2)) ** (1.0 / 4.0),)

                    opacity = float(material.kwargs.get("d", [1.0])[0])
                    if opacity < 1.0:
                        if color_factor is None:
                            color_factor = (1.0, 1.0, 1.0, opacity)
                        else:
                            color_factor = (*color_factor[:3], color_factor[3] * opacity)
                else:
                    gs.raise_exception()

            else:
                # TODO: support vertex/face colors in luisa
                color_factor = tuple(np.array(mesh.visual.main_color, dtype=float) / 255.0)

        else:
            # use white color as default
            color_factor = (1.0, 1.0, 1.0, 1.0)

        color_texture = mu.create_texture(color_image, color_factor, "srgb")
        opacity_texture = None
        if color_texture is not None:
            opacity_texture = color_texture.check_dim(3)
        roughness_texture = mu.create_texture(None, roughness_factor, "linear")

        surface.update_texture(
            color_texture=color_texture,
            opacity_texture=opacity_texture,
            roughness_texture=roughness_texture,
        )
        mesh.visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(mesh.vertices))

        if scale is not None:
            mesh.vertices *= scale

        return cls(
            mesh=mesh,
            surface=surface,
            uvs=uvs,
            convexify=convexify,
            decimate=decimate,
            decimate_face_num=decimate_face_num,
            metadata=metadata,
        )

    @classmethod
    def from_attrs(cls, verts, faces, normals=None, surface=None, uvs=None, scale=None):
        """
        Create a genesis.Mesh from mesh attribtues including vertices, faces, and normals.
        """
        if surface is None:
            surface = gs.surfaces.Default()
        else:
            surface = surface.copy()

        return cls(
            mesh=trimesh.Trimesh(
                vertices=verts * scale if scale is not None else verts,
                faces=faces,
                vertex_normals=normals,
                visual=mu.surface_uvs_to_trimesh_visual(surface, uvs, len(verts)),
                process=False,
            ),
            surface=surface,
            uvs=uvs,
        )

    @classmethod
    def from_morph_surface(cls, morph, surface=None):
        """
        Create a genesis.Mesh from morph and surface options.
        If the morph is a Mesh morph (morphs.Mesh), it could contain multiple submeshes, so we return a list.
        """
        if isinstance(morph, gs.options.morphs.Mesh):
            if morph.file.endswith(("obj", "ply", "stl")):
                meshes = mu.parse_mesh_trimesh(morph.file, morph.group_by_material, morph.scale, surface)

            elif morph.file.endswith(("glb", "gltf")):
                if morph.parse_glb_with_trimesh:
                    meshes = mu.parse_mesh_trimesh(morph.file, morph.group_by_material, morph.scale, surface)
                else:
                    meshes = mu.parse_mesh_glb(morph.file, morph.group_by_material, morph.scale, surface)

            elif hasattr(morph, "files") and len(morph.files) > 0:  # for meshset
                meshes = morph.files
                assert all([isinstance(v, trimesh.Trimesh) for v in meshes])
                meshes = [mu.trimesh_to_mesh(v, morph.scale, surface) for v in meshes]

            else:
                gs.raise_exception(
                    f"File type not supported (yet). Submit a feature request if you need this: {morph.file}."
                )

            return meshes

        else:
            if isinstance(morph, gs.options.morphs.Box):
                tmesh = mu.create_box(extents=morph.size)

            elif isinstance(morph, gs.options.morphs.Cylinder):
                tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)

            elif isinstance(morph, gs.options.morphs.Sphere):
                tmesh = mu.create_sphere(radius=morph.radius)

            else:
                gs.raise_exception()

            return cls.from_trimesh(tmesh, surface=surface)

    def set_color(self, color):
        """
        Set the mesh's color.
        """
        color_texture = gs.textures.ColorTexture(color=tuple(color))
        opacity_texture = color_texture.check_dim(3)
        self._surface.update_texture(color_texture=color_texture, opacity_texture=opacity_texture, force=True)
        self.update_trimesh_visual()

    def update_trimesh_visual(self):
        """
        Update the trimesh obj's visual attributes using its surface and uvs.
        """
        self._mesh.visual = mu.surface_uvs_to_trimesh_visual(self.surface, self.uvs, len(self.verts))

    def apply_transform(self, T):
        """
        Apply a 4x4 transformation matrix to the mesh.
        """
        self._mesh.apply_transform(T)

    def show(self):
        """
        Visualize the mesh using trimesh's built-in viewer.
        """
        return self._mesh.show()

    @property
    def uid(self):
        """
        Return the mesh's uid.
        """
        return self._uid

    @property
    def trimesh(self):
        """
        Return the mesh's trimesh object.
        """
        return self._mesh

    @property
    def is_convex(self):
        """
        Whether the mesh is convex.
        """
        return self._mesh.is_convex

    @property
    def metadata(self):
        """
        Metadata of the mesh.
        """
        return self._metadata

    @property
    def verts(self):
        """
        Vertices of the mesh.
        """
        return self._mesh.vertices

    @verts.setter
    def verts(self, verts):
        """
        Set the vertices of the mesh.
        """
        assert len(verts) == len(self.verts)
        self._mesh.vertices = verts

    @property
    def faces(self):
        """
        Faces of the mesh.
        """
        return self._mesh.faces

    @property
    def normals(self):
        """
        Normals of the mesh.
        """
        return self._mesh.vertex_normals

    @property
    def surface(self):
        """
        Surface of the mesh.
        """
        return self._surface

    @property
    def uvs(self):
        """
        UVs of the mesh.
        """
        return self._uvs

    @property
    def area(self):
        """
        Surface area of the mesh.
        """
        return self._mesh.area

    @property
    def volume(self):
        """
        Volume of the mesh.
        """
        return self._mesh.volume
