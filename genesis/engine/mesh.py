import os
import pickle as pkl
from typing import Any

import fast_simplification
import numpy as np
import trimesh

import genesis as gs
import genesis.utils.mesh as mu
import genesis.utils.gltf as gltf_utils
import genesis.utils.particle as pu
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.utils.misc import redirect_libc_stderr


class Mesh(RBC):
    """
    Genesis's own triangle mesh object.

    This is a wrapper of `trimesh.Trimesh` with some additional features and attributes. The internal trimesh object
    can be accessed via `self.trimesh`.

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
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters
        the original geometry if necessary. 8 does what needs to be done at all costs. Default to 0.
    metadata : dict
        The metadata of the mesh.
    """

    def __init__(
        self,
        mesh,
        surface: Surface | None = None,
        uvs: "np.typing.NDArray | None" = None,
        scale: "np.typing.NDArray | float | None" = None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=0,
        metadata=None,
        is_mesh_zup: bool = True,
    ):
        self._uid = gs.UID()
        self._mesh = mesh  # .copy() FIXME: For some reason forcing copy is causing some tests to fails...
        self._surface = surface
        if uvs is not None:
            uvs = uvs.astype(gs.np_float, copy=False)
        self._uvs = uvs
        self._metadata: dict[str, Any] = metadata or {}
        self._color = np.array([1.0, 1.0, 1.0, 1.0], dtype=gs.np_float)

        # By default, all meshes are considered zup, unless the "FileMorph.file_meshes_are_zup" option was set to False
        self._metadata.setdefault("imported_as_zup", True)

        # By default, all meshes are considered having their original visual
        self._metadata.setdefault("is_visual_overwritten", False)

        if not is_mesh_zup:
            if self._metadata["imported_as_zup"]:
                self._mesh.apply_transform(mu.Y_UP_TRANSFORM.T)
            self._metadata["imported_as_zup"] = False

        if scale is not None:
            scale = np.atleast_1d(np.asarray(scale))
            assert scale.ndim == 1 and scale.size in (1, 3)
            self._mesh.apply_scale(scale)

        if self._surface.requires_uv():  # check uvs here
            if self._uvs is None:
                if "mesh_path" in self._metadata:
                    gs.logger.warning(
                        f"Texture given but asset missing uv info (or failed to load): {self._metadata['mesh_path']}"
                    )
                else:
                    gs.logger.warning("Texture given but asset missing uv info (or failed to load).")
        else:
            self._uvs = None

        if convexify:
            self.convexify()

        if decimate:
            self.decimate(decimate_face_num, decimate_aggressiveness, convexify)

    def convexify(self):
        """
        Convexify the mesh.
        """
        if self._mesh.vertices.shape[0] > 3:
            self._mesh = trimesh.convex.convex_hull(self._mesh)
            self._metadata["convexified"] = True
        self.clear_visuals()

    def decimate(self, decimate_face_num, decimate_aggressiveness, convexify):
        """
        Decimate the mesh.
        """
        if self._mesh.vertices.shape[0] > 3 and len(self._mesh.faces) > decimate_face_num:
            self._mesh.process(validate=True)
            self._mesh = trimesh.Trimesh(
                *fast_simplification.simplify(
                    self._mesh.vertices,
                    self._mesh.faces,
                    target_count=decimate_face_num,
                    agg=decimate_aggressiveness,
                    lossless=(decimate_aggressiveness == 0),
                ),
            )
            self._metadata["decimated"] = True

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
            except (EOFError, ModuleNotFoundError, pkl.UnpicklingError, TypeError, MemoryError):
                gs.logger.info("Ignoring corrupted cache.")

        if not is_cached_loaded:
            # Importing pymeshlab is very slow and not used very often. Let's delay import.
            with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
                import pymeshlab

            gs.logger.info("Remeshing for tetrahedralization...")
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

        self._mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.clear_visuals()

    def tetrahedralize(self, tet_cfg):
        """
        Tetrahedralize the mesh.
        """
        return mu.tetrahedralize_mesh(self._mesh, tet_cfg)

    def particlize(
        self,
        p_size=0.01,
        sampler="random",
    ):
        """
        Sample particles using the mesh volume.
        """
        if "pbs" in sampler:
            return pu.trimesh_to_particles_pbs(self._mesh, p_size, sampler)
        return pu.trimesh_to_particles_simple(self._mesh, p_size, sampler)

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
            mesh=self._mesh.copy(**(dict(include_cache=True) if isinstance(self._mesh, trimesh.Trimesh) else {})),
            surface=self._surface.copy(),
            uvs=self._uvs.copy() if self._uvs is not None else None,
            metadata=self._metadata.copy(),
        )

    @classmethod
    def from_trimesh(
        cls,
        mesh,
        scale=None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=2,
        metadata=None,
        surface=None,
        is_mesh_zup=True,
    ):
        """
        Create a genesis.Mesh from a trimesh.Trimesh object.
        """
        if surface is None:
            surface = gs.surfaces.Default()
            surface.update_texture()
        else:
            surface = surface.copy()

        mesh = mesh.copy(**(dict(include_cache=True) if isinstance(mesh, trimesh.Trimesh) else {}))

        # Always parse uvs if available because roughness and normal map also need uvs.
        # Note that some visual may not have uv, e.g. ColorVisuals.
        uvs = None
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and mesh.visual.uv is not None:
            # Note that 'trimesh' uses uvs starting from top left corner.
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]

        metadata = metadata or {}
        must_update_surface = True
        roughness_factor = None
        color_image = None
        color_factor = None
        opacity = 1.0

        visual = mesh.visual
        if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.defined:
            if visual.kind == "texture":
                material = visual.material

                # TODO: Parsing PBR in obj or not
                # trimesh from .obj file will never use PBR material, but that from .glb file will
                if isinstance(material, trimesh.visual.material.PBRMaterial):
                    if material.baseColorTexture is not None:
                        color_image = mu.PIL_to_array(material.baseColorTexture)
                    if material.baseColorFactor is not None:
                        color_factor = tuple(np.array(material.baseColorFactor, dtype=np.float32) / 255.0)

                    if material.roughnessFactor is not None:
                        roughness_factor = (material.roughnessFactor,)

                elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                    if material.image is not None:
                        color_image = mu.PIL_to_array(material.image)
                    elif material.diffuse is not None:
                        color_factor = tuple(np.array(material.diffuse, dtype=np.float32) / 255.0)

                    if material.glossiness is not None:
                        roughness_factor = (mu.glossiness_to_roughness(material.glossiness),)

                    opacity = float(material.kwargs.get("d", [1.0])[0])
                    if opacity < 1.0:
                        if color_factor is None:
                            color_factor = (1.0, 1.0, 1.0, opacity)
                        else:
                            color_factor = (*color_factor[:3], color_factor[3] * opacity)
                else:
                    gs.raise_exception(f"Unsupported Trimesh material type '{type(material)}'.")
            else:
                # TODO: support vertex/face colors in luisa
                color_factor = tuple(np.array(visual.main_color, dtype=np.float32) / 255.0)
        elif surface.color is not None:
            color_factor = surface.color
            metadata["is_visual_overwritten"] = True
        elif (isinstance(visual, trimesh.visual.color.ColorVisuals) and visual.defined) or (
            isinstance(visual, trimesh.visual.color.VertexColor) and visual.vertex_colors.size > 0
        ):
            # Color is already vertex-based. It is not only necessary to create a new visual.
            must_update_surface = False
        else:
            # use white color as default
            color_factor = (1.0, 1.0, 1.0, 1.0)

        if must_update_surface:
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

        return cls(
            mesh=mesh,
            surface=surface,
            uvs=uvs,
            scale=scale,
            convexify=convexify,
            decimate=decimate,
            decimate_face_num=decimate_face_num,
            decimate_aggressiveness=decimate_aggressiveness,
            metadata=metadata,
            is_mesh_zup=is_mesh_zup,
        )

    @classmethod
    def from_attrs(
        cls, verts, faces, normals=None, surface=None, uvs=None, scale=None, metadata=None, is_mesh_zup=True
    ):
        """
        Create a genesis.Mesh from mesh attributes including vertices, faces, and normals.
        """
        if surface is None:
            surface = gs.surfaces.Default()

        metadata = metadata or {}
        metadata["is_visual_overwritten"] = metadata.get("is_visual_overwritten", False) or (surface.color is not None)
        visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(verts))

        tmesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            visual=visual,
            process=False,
        )

        return cls(
            mesh=tmesh,
            surface=surface,
            uvs=uvs,
            scale=scale,
            metadata=metadata,
            is_mesh_zup=is_mesh_zup,
        )

    @classmethod
    def from_morph_surface(cls, morph, surface=None):
        """
        Create a genesis.Mesh from morph and surface options.

        If the morph is a Mesh morph (morphs.Mesh), it could contain multiple sub-meshes, so we return a list.
        """
        if isinstance(morph, gs.options.morphs.Mesh):
            if morph.is_format(gs.options.morphs.MESH_FORMATS):
                if morph.is_format(gs.options.morphs.GLTF_FORMATS):
                    meshes = gltf_utils.parse_mesh_glb(
                        morph.file, morph.group_by_material, morph.scale, morph.file_meshes_are_zup, surface
                    )
                else:
                    meshes = mu.parse_mesh_trimesh(
                        morph.file, morph.group_by_material, morph.scale, morph.file_meshes_are_zup, surface
                    )
            elif isinstance(morph, gs.options.morphs.MeshSet):
                assert all(isinstance(mesh, trimesh.Trimesh) for mesh in morph.files)
                meshes = [mu.trimesh_to_mesh(mesh, morph.scale, surface) for mesh in morph.files]
            else:
                gs.raise_exception(f"File type not supported: {morph.file}")

            return meshes

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
        self._color = color
        color_texture = gs.textures.ColorTexture(color=tuple(color))
        opacity_texture = color_texture.check_dim(3)
        self._surface.update_texture(color_texture=color_texture, opacity_texture=opacity_texture, force=True)
        self.update_trimesh_visual()

    def update_trimesh_visual(self):
        """
        Update the trimesh obj's visual attributes using its surface and uvs.
        """
        self._mesh.visual = mu.surface_uvs_to_trimesh_visual(self.surface, self.uvs, len(self.verts))
        self._metadata["is_visual_overwritten"] = True

    def apply_transform(self, T):
        """
        Apply a 4x4 transformation matrix (translation on the right column) to the mesh.
        """
        self._mesh.apply_transform(T)

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
    def is_convex(self) -> bool:
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
