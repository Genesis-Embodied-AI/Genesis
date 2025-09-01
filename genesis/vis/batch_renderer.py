import math

import numpy as np
from PIL import Image
import torch
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.color import ColorVisuals

import genesis as gs
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array
from genesis.constants import IMAGE_TYPE

from .rasterizer_context import SegmentationColorMap

try:
    from gs_madrona.renderer_gs import MadronaBatchRendererAdapter, GeomRetriever
except ImportError as e:
    gs.raise_exception_from("Madrona batch renderer is only supported on Linux x86-64.", e)


TYPES = (IMAGE_TYPE.RGB, IMAGE_TYPE.DEPTH, IMAGE_TYPE.NORMAL, IMAGE_TYPE.SEGMENTATION)  # order of RenderOption


def transform_camera_quat(quat):
    # quat for Madrona needs to be transformed to y-forward
    w, x, y, z = torch.unbind(quat, dim=-1)
    return torch.stack([x + w, x - w, y - z, y + z], dim=-1) / math.sqrt(2.0)


class GenesisGeomRetriever(GeomRetriever):
    def __init__(self, rigid_solver, seg_level):
        self.rigid_solver = rigid_solver
        self.seg_color_map = SegmentationColorMap()
        self.seg_level = seg_level
        self.geom_idxc = None

        self.default_geom_group = 2
        self.default_enabled_geom_groups = np.array([self.default_geom_group], dtype=np.int32)
        self.num_textures_per_material = 10  # Madrona allows up to 10 textures per material, should not change.

    def build(self):
        self.n_vgeoms = self.rigid_solver.n_vgeoms
        self.geom_idxc = []
        vgeoms = self.rigid_solver.vgeoms
        for vgeom in vgeoms:
            seg_key = self.get_seg_key(vgeom)
            seg_idxc = self.seg_color_map.seg_key_to_idxc(seg_key)
            self.geom_idxc.append(seg_idxc)
        self.geom_idxc = np.array(self.geom_idxc, dtype=np.uint32)
        self.seg_color_map.generate_seg_colors()

    def get_seg_key(self, vgeom):
        if self.seg_level == "geom":
            return (vgeom.entity.idx, vgeom.link.idx, vgeom.idx)
        elif self.seg_level == "link":
            return (vgeom.entity.idx, vgeom.link.idx)
        elif self.seg_level == "entity":
            return vgeom.entity.idx
        else:
            gs.raise_exception(f"Unsupported segmentation level: {self.seg_level}")

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_meshes_static(self):
        args = {}
        vgeoms = self.rigid_solver.vgeoms

        # Retrieve geom data
        mesh_vertices = self.rigid_solver.vverts_info.init_pos.to_numpy()
        mesh_faces = self.rigid_solver.vfaces_info.vverts_idx.to_numpy()
        mesh_vertex_offsets = self.rigid_solver.vgeoms_info.vvert_start.to_numpy()
        mesh_face_starts = self.rigid_solver.vgeoms_info.vface_start.to_numpy()
        mesh_face_ends = self.rigid_solver.vgeoms_info.vface_end.to_numpy()
        for i in range(self.n_vgeoms):
            mesh_faces[mesh_face_starts[i] : mesh_face_ends[i]] -= mesh_vertex_offsets[i]

        geom_data_ids = []
        for vgeom in vgeoms:
            seg_key = self.get_seg_key(vgeom)
            seg_id = self.seg_color_map.seg_key_to_idxc(seg_key)
            geom_data_ids.append(seg_id)

        args["mesh_vertices"] = mesh_vertices
        args["mesh_faces"] = mesh_faces
        args["mesh_vertex_offsets"] = mesh_vertex_offsets
        args["mesh_face_offsets"] = mesh_face_starts
        args["geom_types"] = np.full((self.n_vgeoms,), 7, dtype=np.int32)  # 7 stands for mesh
        args["geom_groups"] = np.full((self.n_vgeoms,), self.default_geom_group, dtype=np.int32)
        args["geom_data_ids"] = np.arange(self.n_vgeoms, dtype=np.int32)
        args["geom_sizes"] = np.ones((self.n_vgeoms, 3), dtype=np.float32)
        args["enabled_geom_groups"] = self.default_enabled_geom_groups

        # Retrieve material data
        num_materials = 0
        total_uv_size = 0
        total_texture_size = 0
        geom_mat_ids = []
        geom_uv_sizes = []
        geom_uv_offsets = []
        geom_rgbas = []

        mat_uv_data = []
        mat_texture_widths = []
        mat_texture_heights = []
        mat_texture_nchans = []
        mat_texture_offsets = []
        mat_texture_data = []
        mat_texture_ids = []
        mat_rgbas = []

        for vgeom in vgeoms:
            visual = vgeom.get_trimesh().visual
            if isinstance(visual, TextureVisuals):
                uv_size = visual.uv.shape[0]
                geom_mat_ids.append(num_materials)
                geom_uv_sizes.append(uv_size)
                geom_uv_offsets.append(total_uv_size)
                geom_rgbas.append(np.empty((4,), dtype=np.float32))

                texture_width = visual.material.image.width
                texture_height = visual.material.image.height
                texture_nchans = 4 if visual.material.image.mode == "RGBA" else 3
                texture_size = texture_width * texture_height * texture_nchans
                texture_ids = np.full((self.num_textures_per_material,), -1, np.int32)
                texture_ids[0] = num_materials

                mat_uv_data.append(visual.uv.astype(np.float32))
                mat_texture_widths.append(texture_width)
                mat_texture_heights.append(texture_height)
                mat_texture_nchans.append(texture_nchans)
                mat_texture_offsets.append(total_texture_size)
                mat_texture_data.append(
                    np.array(
                        visual.material.image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).getdata(), dtype=np.uint8
                    ).flatten()
                )
                mat_texture_ids.append(texture_ids)
                mat_rgbas.append(visual.material.diffuse.astype(np.float32) / 255.0)

                num_materials += 1
                total_uv_size += uv_size
                total_texture_size += texture_size
            else:
                geom_mat_ids.append(-1)
                geom_uv_sizes.append(0)
                geom_uv_offsets.append(-1)
                if isinstance(visual, ColorVisuals):
                    geom_rgbas.append(visual.main_color.astype(np.float32) / 255.0)
                else:
                    geom_rgbas.append(np.empty((4,), dtype=np.float32))

        args["geom_mat_ids"] = np.array(geom_mat_ids, np.int32)
        args["mesh_texcoord_num"] = np.array(geom_uv_sizes, np.int32)
        args["mesh_texcoord_offsets"] = np.array(geom_uv_offsets, np.int32)
        args["geom_rgba"] = np.stack(geom_rgbas, axis=0)

        args["mesh_texcoords"] = np.concatenate(mat_uv_data, axis=0) if mat_uv_data else np.empty((0, 2), np.float32)
        args["tex_widths"] = np.array(mat_texture_widths, np.int32)
        args["tex_heights"] = np.array(mat_texture_heights, np.int32)
        args["tex_nchans"] = np.array(mat_texture_nchans, np.int32)
        args["tex_offsets"] = np.array(mat_texture_offsets, np.int32)
        args["tex_data"] = np.concatenate(mat_texture_data, axis=0) if mat_texture_data else np.array([], np.uint8)
        args["mat_tex_ids"] = (
            np.stack(mat_texture_ids, axis=0)
            if mat_texture_ids
            else np.empty((0, self.num_textures_per_material), np.int32)
        )
        args["mat_rgba"] = np.stack(mat_rgbas, axis=0) if mat_rgbas else np.empty((0, 4), np.float32)

        return args

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_property_torch(self, num_worlds):
        geom_rgb_torch = self.rigid_solver.vgeoms_info.color.to_torch()
        geom_rgb_int = (geom_rgb_torch * 255).to(torch.int32)
        geom_rgb_uint = (geom_rgb_int[:, 0] << 16) | (geom_rgb_int[:, 1] << 8) | geom_rgb_int[:, 2]
        geom_rgb = geom_rgb_uint.unsqueeze(0).repeat(num_worlds, 1)

        geom_mat_ids = torch.full((self.n_vgeoms,), -1, dtype=torch.int32, device=gs.device)
        geom_mat_ids = geom_mat_ids.unsqueeze(0).repeat(num_worlds, 1)

        geom_sizes = torch.ones((self.n_vgeoms, 3), dtype=torch.float32, device=gs.device)
        geom_sizes = geom_sizes.unsqueeze(0).repeat(num_worlds, 1, 1)
        return geom_mat_ids, geom_rgb, geom_sizes

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_state_torch(self):
        geom_pos = self.rigid_solver.vgeoms_state.pos.to_torch()
        geom_rot = self.rigid_solver.vgeoms_state.quat.to_torch()
        geom_pos = geom_pos.transpose(0, 1).contiguous()
        geom_rot = geom_rot.transpose(0, 1).contiguous()
        return geom_pos, geom_rot


class Light:
    def __init__(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._pos = pos
        self._dir = tuple(dir / np.linalg.norm(dir))  # Converting list of arrays to tensor is inefficient.
        self._color = color
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff
        self._attenuation = attenuation

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    @property
    def color(self):
        return self._color

    @property
    def intensity(self):
        return self._intensity

    @property
    def directional(self):
        return self._directional

    @property
    def castshadow(self):
        return self._castshadow

    @property
    def cutoffRad(self):
        return math.radians(self._cutoff)

    @property
    def cutoffDeg(self):
        return self._cutoff

    @property
    def attenuation(self):
        return self._attenuation


class BatchRenderer(RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer, renderer_options, vis_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._use_rasterizer = renderer_options.use_rasterizer
        self._renderer = None
        self._geom_retriever = GenesisGeomRetriever(self._visualizer.scene.rigid_solver, vis_options.segmentation_level)
        self._data_cache = {}
        self._t = -1

    def add_light(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._lights.append(Light(pos, dir, color, intensity, directional, castshadow, cutoff, attenuation))

    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if len(self._visualizer._cameras) == 0:
            raise ValueError("No cameras to render")

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRenderer requires CUDA backend.")

        self._cameras = gs.List([camera for camera in self._visualizer._cameras if not camera.debug])
        self._geom_retriever.build()
        n_envs = max(self._visualizer.scene.n_envs, 1)
        res = self._cameras[0].res  # Madrona uses resolution of first (non-debug) camera
        gpu_id = gs.device.index if gs.device.index is not None else 0

        # Cameras
        n_cameras = len(self._cameras)
        cameras_pos = torch.stack([camera.get_pos() for camera in self._cameras], dim=1)
        cameras_quat = torch.stack([camera.get_quat() for camera in self._cameras], dim=1)
        cameras_quat = transform_camera_quat(cameras_quat)
        cameras_fov = torch.tensor([camera.fov for camera in self._cameras], dtype=torch.float32, device=gs.device)
        cameras_near = torch.tensor([camera.near for camera in self._cameras], dtype=torch.float32, device=gs.device)
        cameras_far = torch.tensor([camera.far for camera in self.cameras], dtype=torch.float32, device=gs.device)

        # Build taichi arrays to store light properties once. If later we need to support dynamic lights, we should
        # consider storing light properties as taichi fields in Genesis.
        n_lights = len(self._lights)
        if n_lights:
            light_pos = torch.tensor([light.pos for light in self._lights], dtype=torch.float32, device=gs.device)
            light_dir = torch.tensor([light.dir for light in self._lights], dtype=torch.float32, device=gs.device)
            light_rgb = torch.tensor([light.color for light in self._lights], dtype=torch.float32, device=gs.device)
        else:
            light_pos = torch.empty((0, 3), dtype=torch.float32, device=gs.device)
            light_dir = torch.empty((0, 3), dtype=torch.float32, device=gs.device)
            light_rgb = torch.empty((0, 3), dtype=torch.float32, device=gs.device)
        light_directional = torch.tensor(
            [light.directional for light in self._lights], dtype=torch.bool, device=gs.device
        )
        light_castshadow = torch.tensor(
            [light.castshadow for light in self._lights], dtype=torch.bool, device=gs.device
        )
        light_cutoff = torch.tensor([light.cutoffRad for light in self._lights], dtype=torch.float32, device=gs.device)
        light_attenuation = torch.tensor(
            [light.attenuation for light in self._lights], dtype=torch.float32, device=gs.device
        )
        light_intensity = torch.tensor(
            [light.intensity for light in self._lights], dtype=torch.float32, device=gs.device
        )

        self._renderer = MadronaBatchRendererAdapter(
            self._geom_retriever,
            gpu_id,
            n_envs,
            n_cameras,
            n_lights,
            cameras_fov,
            cameras_near,
            cameras_far,
            *res,
            False,
            self._use_rasterizer,
        )
        self._renderer.init(
            cameras_pos,
            cameras_quat,
            light_pos,
            light_dir,
            light_rgb,
            light_directional,
            light_castshadow,
            light_cutoff,
            light_attenuation,
            light_intensity,
        )

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False, force_render=False):
        """
        Render all cameras in the batch.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render the rgb image.
        depth : bool, optional
            Whether to render the depth image.
        segmentation : bool, optional
            Whether to render the segmentation image.
        normal : bool, optional
            Whether to render the normal image.
        antialiasing : bool, optional
            Whether to apply anti-aliasing.
        force_render : bool, optional
            Whether to force render the scene.

        Returns
        -------
        rgb_arr : tuple of arrays
            The sequence of rgb images associated with each camera.
        depth_arr : tuple of arrays
            The sequence of depth images associated with each camera.
        segmentation_arr : tuple of arrays
            The sequence of segmentation images associated with each camera.
        normal_arr : tuple of arrays
            The sequence of normal images associated with each camera.
        """

        # Clear cache if requested or necessary
        if force_render or self._t < self._visualizer.scene.t:
            self._data_cache.clear()

        # Fetch available cached data
        req = [rgb, depth, segmentation, normal]
        cache_key = (antialiasing,)
        cached = [self._data_cache.get((t, cache_key), None) for t in range(IMAGE_TYPE.NUM_TYPES)]
        need = [(req[t] and cached[t] is None) for t in range(IMAGE_TYPE.NUM_TYPES)]

        # Early return if everything requested is already cached
        if not any(need):
            return tuple(cached[t] if req[t] else None for t in range(IMAGE_TYPE.NUM_TYPES))

        # Update scene render only what’s needed (flags still passed to renderer)
        self.update_scene()
        cameras_pos = torch.stack([camera.get_pos() for camera in self._cameras], dim=1)
        cameras_quat = torch.stack([camera.get_quat() for camera in self._cameras], dim=1)
        cameras_quat = transform_camera_quat(cameras_quat)
        render_flags = np.array(
            (
                *(need[t] for t in TYPES),
                antialiasing,
            ),
            dtype=np.uint32,
        )

        # Post-processing:
        # * Remove alpha channel from RGBA
        # * Squeeze env and channel dims if necessary
        # * Split along camera dim
        rendered = self._renderer.render(cameras_pos, cameras_quat, render_flags)
        rendered = [
            tensor_to_array(rendered[t][..., :3].flip(-1).squeeze(-1).swapaxes(0, 1)) if need[t] else None
            for t in range(IMAGE_TYPE.NUM_TYPES)
        ]

        # convert center distance depth to plane distance
        if not self._use_rasterizer:
            if need[IMAGE_TYPE.DEPTH]:
                depth_centers = rendered[IMAGE_TYPE.DEPTH]
                for i, camera in enumerate(self._cameras):
                    depth_centers[i] = camera.distance_center_to_plane(depth_centers[i])

        # convert seg geom idx to seg_idxc
        if need[IMAGE_TYPE.SEGMENTATION]:
            seg_geoms = rendered[IMAGE_TYPE.SEGMENTATION]
            mask = seg_geoms != -1
            seg_geoms[mask] = self._geom_retriever.geom_idxc[seg_geoms[mask]]
            seg_geoms[~mask] = 0

        for t, data in enumerate(rendered):
            if data is not None:
                if self._visualizer.scene.n_envs == 0:
                    data = data.squeeze(1)
                rendered[t] = tuple(data)

        self._t = self._visualizer.scene.t
        for t in TYPES:
            if need[t]:
                self._data_cache[(t, cache_key)] = rendered[t]

        # Return in the required order, or None if not requested
        return tuple(rendered[t] if need[t] else cached[t] if req[t] else None for t in range(IMAGE_TYPE.NUM_TYPES))

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self._geom_retriever.seg_color_map.colorize_seg_idxc_arr(seg_idxc_arr)

    def destroy(self):
        self._lights.clear()
        self._data_cache.clear()
        if self._renderer is not None:
            del self._renderer.madrona
            self._renderer = None

    def reset(self):
        self._t = -1

    @property
    def lights(self):
        return self._lights

    @property
    def cameras(self):
        return self._cameras

    @property
    def seg_idxc_map(self):
        return self._geom_retriever.seg_color_map.idxc_map
