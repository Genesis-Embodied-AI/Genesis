"""
Apollo SDK Scene Exporter

This module provides functionality to export Genesis scenes directly to Apollo SDK SceneAsset format.
Unlike the JSON-based SceneDescription exporter, this uses the Apollo SDK Python bindings directly.
"""

import genesis as gs
import numpy as np
import os
import math
from pathlib import Path
from typing import Optional
from gs_apollo import apollo_py_sdk as ap

# Coordinate conversion helpers (Z-up to Y-up)
def _to_numpy(val):
    """Convert value to numpy, handling torch tensors (including CUDA tensors)."""
    if hasattr(val, 'cpu'):
        # It's a torch tensor, move to CPU first
        return val.cpu().numpy()
    elif hasattr(val, 'numpy'):
        # It has a numpy method
        return val.numpy()
    elif isinstance(val, np.ndarray):
        return val
    else:
        # Try to convert directly
        return np.array(val)

def _pos_to_y_up(pos):
    """Convert position from Z-up to Y-up: (X, Y, Z) -> (X, Z, -Y)"""
    pos = _to_numpy(pos)
    return np.array([pos[0], pos[2], -pos[1]])

def _quat_multiply(q1, q2):
    """Multiply two quaternions in WXYZ format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_conjugate(q):
    """Return the conjugate of a quaternion in WXYZ format."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def _quat_to_y_up(quat, convert=True):
    """
    Convert quaternion from Z-up to Y-up by left-multiplying -90° about X.
    Input is (w, x, y, z), output is (x, y, z, w) for Apollo.
    If convert=False, just reorder to (x, y, z, w).
    """
    quat = _to_numpy(quat)
    if convert:
        w, x, y, z = quat
        # Apply -90° rotation about X axis to convert from Z-up to Y-up
        quat_rotated = np.array([x + w, x - w, z + y, z - y]) / math.sqrt(2.0)  # WXYZ

        # Convert to XYZW for Apollo
        return np.array([quat_rotated[1], quat_rotated[2], quat_rotated[3], quat_rotated[0]])
    else:
        # Just reorder from WXYZ to XYZW
        return np.array([quat[1], quat[2], quat[3], quat[0]])

def _scale_to_y_up(scale):
    """Convert scale from Z-up to Y-up: (X, Y, Z) -> (X, Z, Y)"""
    scale = _to_numpy(scale)
    return np.array([scale[0], scale[2], scale[1]])

def _dir_to_y_up(dir):
    """Convert direction from Z-up to Y-up: (X, Y, Z) -> (X, Z, Y)"""
    dir = _to_numpy(dir)
    return np.array([dir[0], dir[2], dir[1]])

class ApolloSceneExporter:
    def __init__(self):
        self._scene = None
        self._scene_asset = None
        self._asset_dir = None
        self._export_dir = None  # Directory where the scene file will be exported

    def generate_from_scene(self, scene: gs.Scene, export_path: Optional[str] = None) -> ap.SceneAsset:
        assert scene.is_built, "Scene must be built before exporting"

        self._scene = scene
        self._asset_dir = Path(gs.utils.get_assets_dir())

        # Determine export directory for relative path calculation
        if export_path is not None:
            self._export_dir = Path(export_path).parent.resolve()
        else:
            self._export_dir = Path.cwd()

        # Create a new SceneAsset
        self._scene_asset = ap.SceneAsset()

        # Set root folder to empty string (paths will be relative to export location)
        self._scene_asset.rootFolder = ""

        # Export entities (instances)
        self._export_entities()

        # Export cameras
        self._export_cameras()

        # Export lights
        self._export_lights()

        return self._scene_asset
    
    def _export_entities(self):
        """Export all rigid entities as instances."""
        if not hasattr(self._scene, 'rigid_solver') or not self._scene.rigid_solver.is_active:
            return

        entities = self._scene.rigid_solver.entities
        if not entities:
            return

        # Count total instances needed (some entities have multiple vgeoms)
        total_instances = 0
        for entity in entities:
            if self._should_export_at_geom_level(entity.morph):
                # For URDF/MJCF, export each vgeom with a mesh as a separate instance
                for vgeom in entity.vgeoms:
                    if "mesh_path" in vgeom.metadata:
                        total_instances += 1
            else:
                # For Mesh/Primitive, export as single instance
                total_instances += 1

        if total_instances == 0:
            return

        # Resize the instance array
        self._scene_asset.instance_resize(total_instances)

        instance_idx = 0
        for entity in entities:
            if self._should_export_at_geom_level(entity.morph):
                # Export URDF/MJCF as multiple instances (one per vgeom with mesh)
                instance_idx = self._export_entity_vgeoms(entity, instance_idx)
            else:
                # Export Mesh/Primitive as single instance
                instance_idx = self._export_entity_single(entity, instance_idx)

    def _should_export_at_geom_level(self, morph: gs.morphs.Morph) -> bool:
        """Check if entity should be exported at geom level (URDF/MJCF) or as single instance."""
        return not isinstance(morph, (gs.morphs.Mesh, gs.morphs.Primitive, gs.morphs.Plane))

    def _get_relative_path(self, absolute_path: Path) -> str:
        """Convert an absolute path to a path relative to the export directory."""
        try:
            return os.path.relpath(absolute_path, self._export_dir)
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            # Fall back to absolute path
            return str(absolute_path)

    def _export_entity_single(self, entity, instance_idx: int) -> int:
        """Export a single entity (Mesh/Primitive/Plane) as one instance."""
        instance = self._scene_asset.get_instance(instance_idx)

        # Generate UUID
        instance.uuid = ap.generate_uuid()

        # Set transform with coordinate conversion (Z-up to Y-up)
        pos = entity.morph.pos
        quat = entity.morph.quat

        # Convert position: (X, Y, Z) -> (X, Z, -Y)
        pos_y_up = _pos_to_y_up(pos)
        instance.position = ap.float3(float(pos_y_up[0]), float(pos_y_up[1]), float(pos_y_up[2]))

        # Convert rotation: only apply quaternion conversion for Mesh entities
        convert_quat = isinstance(entity.morph, gs.morphs.Mesh)
        quat_y_up = _quat_to_y_up(quat, convert=convert_quat)
        instance.rotation = ap.quaternion(float(quat_y_up[0]), float(quat_y_up[1]), float(quat_y_up[2]), float(quat_y_up[3]))  # Apollo quaternion constructor is (x, y, z, w)

        # Set scale with coordinate conversion
        scale = self._get_entity_scale(entity.morph)
        scale_y_up = _scale_to_y_up(scale)
        instance.scale = ap.float3(float(scale_y_up[0]), float(scale_y_up[1]), float(scale_y_up[2]))

        # Set type and mesh URI
        if isinstance(entity.morph, gs.morphs.Mesh):
            instance.type = ap.EInstanceType.Mesh
            # Get mesh file path - resolve to absolute first, then make relative to export dir
            mesh_path = Path(entity.morph.file)
            if not mesh_path.is_absolute():
                mesh_path = self._asset_dir / mesh_path
            instance.mesh_uri = self._get_relative_path(mesh_path)
        elif isinstance(entity.morph, gs.morphs.Plane):
            # Check Plane before Primitive since Plane is a subclass of Primitive
            instance.type = ap.EInstanceType.Primitive
            instance.primitive_type = ap.EPrimitiveType.Plane
        elif isinstance(entity.morph, gs.morphs.Primitive):
            instance.type = ap.EInstanceType.Primitive
            instance.primitive_type = self._get_primitive_type(entity.morph)

        # Set material override from entity.surface
        self._set_material_override(instance, entity)

        # Set processing flags
        instance.enabled = True
        instance.dynamic = not entity.morph.fixed if hasattr(entity.morph, 'fixed') else False
        # Keep smooth as default (false) - don't override unless needed
        instance.smooth = False

        # Write back to scene asset
        self._scene_asset.set_instance(instance_idx, instance)

        return instance_idx + 1

    def _export_entity_vgeoms(self, entity, instance_idx: int) -> int:
        """Export URDF/MJCF entity as multiple instances (one per vgeom with mesh)."""
        # TODO: Implement vgeom export for URDF/MJCF entities
        # This requires getting transform and mesh path from each vgeom
        for vgeom in entity.vgeoms:
            if "mesh_path" not in vgeom.metadata:
                continue

            instance = self._scene_asset.get_instance(instance_idx)

            # Generate UUID
            instance.uuid = ap.generate_uuid()

            # Get vgeom transform
            # TODO: Get actual vgeom position and rotation from scene state
            instance.position = ap.float3(0.0, 0.0, 0.0)
            instance.rotation = ap.quaternion(0.0, 0.0, 0.0, 1.0)
            instance.scale = ap.float3(1.0, 1.0, 1.0)

            # Set mesh URI
            instance.type = ap.EInstanceType.Mesh
            mesh_path = Path(vgeom.metadata["mesh_path"])
            # Resolve to absolute path first, then make relative to export dir
            if not mesh_path.is_absolute():
                mesh_path = self._asset_dir / mesh_path
            instance.mesh_uri = self._get_relative_path(mesh_path)

            # Set processing flags
            instance.enabled = True
            instance.dynamic = False
            instance.smooth = True

            # Write back to scene asset
            self._scene_asset.set_instance(instance_idx, instance)

            instance_idx += 1

        return instance_idx

    def _get_entity_scale(self, morph: gs.morphs.Morph) -> tuple:
        """Get scale from morph, handling primitives and meshes differently."""
        if isinstance(morph, gs.morphs.Primitive):
            # For primitives, use their specific dimensions
            if isinstance(morph, gs.morphs.Box):
                return (morph.size[0], morph.size[1], morph.size[2])
            elif isinstance(morph, gs.morphs.Sphere):
                return (morph.radius, morph.radius, morph.radius)
            elif isinstance(morph, gs.morphs.Cylinder):
                return (morph.radius, morph.radius, morph.height)
            elif isinstance(morph, gs.morphs.Plane):
                return (morph.plane_size[0], morph.plane_size[1], 1.0)
            else:
                return (1.0, 1.0, 1.0)
        elif hasattr(morph, 'scale'):
            # For meshes and other morphs, use the scale attribute
            scale = morph.scale
            if isinstance(scale, (int, float)):
                return (scale, scale, scale)
            elif isinstance(scale, (list, tuple)):
                if len(scale) == 3:
                    return tuple(scale)
                else:
                    return (scale[0], scale[0], scale[0])
        return (1.0, 1.0, 1.0)

    def _get_primitive_type(self, morph: gs.morphs.Primitive) -> ap.EPrimitiveType:
        """Convert Genesis primitive type to Apollo primitive type."""
        if isinstance(morph, gs.morphs.Plane):
            return ap.EPrimitiveType.Plane
        elif isinstance(morph, gs.morphs.Sphere):
            return ap.EPrimitiveType.Sphere
        elif isinstance(morph, gs.morphs.Box):
            return ap.EPrimitiveType.Box
        elif isinstance(morph, gs.morphs.Cylinder):
            return ap.EPrimitiveType.Cylinder
        else:
            # Default to box
            return ap.EPrimitiveType.Box

    def _set_material_override(self, instance: ap.InstanceAsset, entity):
        """Set material override properties from entity surface."""
        # For primitives, the surface with texture is on the vgeom's vmesh, not on the entity
        surface = None
        if isinstance(entity.morph, gs.morphs.Primitive):
            # Check if there's a vgeom with a vmesh that has a surface
            if hasattr(entity, 'vgeoms') and entity.vgeoms:
                vgeom = entity.vgeoms[0]  # Primitives have one vgeom
                if hasattr(vgeom, 'vmesh') and vgeom.vmesh is not None:
                    surface = vgeom.vmesh.surface

        # Fall back to entity.surface if not found on vgeom
        if surface is None:
            surface = entity.surface

        if surface is None:
            return

        mat_override = instance.matOverride

        # Set albedo color if available
        if hasattr(surface, 'color') and surface.color is not None:
            color = _to_numpy(surface.color)
            if len(color) >= 3:
                # Set albedo color (RGB or RGBA)
                alpha = color[3] if len(color) >= 4 else 1.0
                mat_override.albedoColor = ap.float4(float(color[0]), float(color[1]), float(color[2]), float(alpha))
                mat_override.set_properties(ap.EMaterialProperty.AlbedoColor)

        # Set diffuse texture if available
        if hasattr(surface, 'diffuse_texture') and surface.diffuse_texture is not None:
            texture = surface.diffuse_texture
            if hasattr(texture, 'input_image_path') and texture.input_image_path is not None:
                # Get texture path - resolve to absolute first, then make relative to export dir
                texture_path = Path(texture.input_image_path)
                if not texture_path.is_absolute():
                    texture_path = self._asset_dir / texture_path
                relative_texture_path = self._get_relative_path(texture_path)
                mat_override.albedoTexture = relative_texture_path
                mat_override.set_properties(ap.EMaterialProperty.AlbedoTexture)

                # Set UV scale for planes (100x100 for checker texture)
                if isinstance(entity.morph, gs.morphs.Plane):
                    # Use plane size for UV scale
                    plane_size = entity.morph.plane_size
                    mat_override.uvScale = ap.float2(float(plane_size[0]), float(plane_size[1]))
                    mat_override.set_properties(ap.EMaterialProperty.UVScale)

    def _export_cameras(self):
        """Export all cameras."""
        if not hasattr(self._scene, 'visualizer') or self._scene.visualizer is None:
            return
            
        cameras = self._scene.visualizer.cameras
        if not cameras:
            return
            
        # Resize the camera array
        self._scene_asset.camera_resize(len(cameras))
        
        for idx, camera in enumerate(cameras):
            cam_asset = self._scene_asset.get_camera(idx)

            # Generate UUID
            cam_asset.uuid = ap.generate_uuid()

            # Set camera transform with coordinate conversion
            # This follows the same approach as the old scene_exporter._generate_camera_desc()
            cam_pos = camera._initial_pos
            cam_lookat = camera._initial_lookat
            cam_up = camera._initial_up

            # Get camera transform matrix in Genesis Z-up space
            if camera._initial_transform is not None:
                cam_transform = camera._initial_transform
            else:
                cam_transform = gs.utils.geom.pos_lookat_up_to_T(cam_pos, cam_lookat, cam_up)

            # Extract quaternion from rotation matrix
            cam_transform_np = _to_numpy(cam_transform)
            R_genesis = cam_transform_np[:3, :3]
            cam_quat = gs.utils.geom.R_to_quat(R_genesis)

            # Convert to Y-up coordinate system (same as old exporter)
            pos_y_up = _pos_to_y_up(cam_pos)
            quat_y_up = _quat_to_y_up(cam_quat, convert=True)

            cam_asset.position = ap.float3(float(pos_y_up[0]), float(pos_y_up[1]), float(pos_y_up[2]))
            cam_asset.rotation = ap.quaternion(float(quat_y_up[0]), float(quat_y_up[1]), float(quat_y_up[2]), float(quat_y_up[3]))  # Apollo quaternion constructor is (x, y, z, w)
            
            # Set camera parameters
            cam_asset.resolution = ap.uint2(int(camera.res[0]), int(camera.res[1]))
            # Convert FOV from degrees (Genesis) to radians (Apollo internal format)
            cam_asset.fovY = float(np.deg2rad(camera.fov))
            cam_asset.nearPlane = float(camera.near)
            cam_asset.farPlane = float(camera.far)

            # Set camera parameters - use Apollo defaults, not Genesis computed values
            # Genesis focal_len is computed in pixels, Apollo expects physical focal length in mm
            cam_asset.aperture = 2.0  # Fixed value for reference scene
            cam_asset.focalLength = 10.0  # Fixed physical focal length in mm
            cam_asset.spp = 32  # Lower than Genesis default (256)
            cam_asset.denoise = True
            # Note: Using getattr to access 'None' enum value since it's a Python keyword
            cam_asset.toneMapper = getattr(ap.EToneMapper, 'None')
            cam_asset.antiAliasing = getattr(ap.EAntiAliasing, 'None')
            
            # Write back to scene asset
            self._scene_asset.set_camera(idx, cam_asset)
    
    def _export_lights(self):
        """Export all lights."""
        # Check if we have apollo_renderer with lights
        if not hasattr(self._scene.visualizer, 'apollo_renderer') or self._scene.visualizer.apollo_renderer is None:
            return

        lights = self._scene.visualizer.apollo_renderer.lights
        if not lights:
            return

        # Resize the light array
        self._scene_asset.light_resize(len(lights))

        for idx, light in enumerate(lights):
            light_asset = self._scene_asset.get_light(idx)

            # Generate UUID
            light_asset.uuid = ap.generate_uuid()

            # Set common properties
            light_asset.color = ap.float3(float(light.color[0]), float(light.color[1]), float(light.color[2]))
            light_asset.intensity = float(light.intensity)
            light_asset.shadow = bool(light.castshadow)
            light_asset.unit = ap.ELightUnit.Lux  # Default unit

            # Set type-specific properties based on light type with coordinate conversion
            if light.directional:
                # Directional light
                light_asset.type = ap.ELightType.Directional
                dir_y_up = _dir_to_y_up(light.dir)
                light_asset.directional_direction = ap.float3(
                    float(dir_y_up[0]),
                    float(dir_y_up[1]),
                    float(dir_y_up[2])
                )
            elif light.cutoffDeg is not None and light.cutoffDeg > 0.0 and light.cutoffDeg < 90.0:
                # Spot light
                light_asset.type = ap.ELightType.Spot
                pos_y_up = _pos_to_y_up(light.pos)
                dir_y_up = _dir_to_y_up(light.dir)
                light_asset.spot_position = ap.float3(
                    float(pos_y_up[0]),
                    float(pos_y_up[1]),
                    float(pos_y_up[2])
                )
                light_asset.spot_direction = ap.float3(
                    float(dir_y_up[0]),
                    float(dir_y_up[1]),
                    float(dir_y_up[2])
                )
                light_asset.spot_outerAngle = float(light.cutoffDeg)
                light_asset.spot_innerAngle = 0.0
                light_asset.spot_falloff = 1.0
                light_asset.spot_radius = 100.0
                light_asset.spot_attenuation = float(light.attenuation) if light.attenuation > 0 else 1.8
            else:
                # Point light
                light_asset.type = ap.ELightType.Point
                pos_y_up = _pos_to_y_up(light.pos)
                light_asset.point_position = ap.float3(
                    float(pos_y_up[0]),
                    float(pos_y_up[1]),
                    float(pos_y_up[2])
                )
                light_asset.point_radius = 100.0
                light_asset.point_attenuation = float(light.attenuation) if light.attenuation > 0 else 1.8

            # Write back to scene asset
            self._scene_asset.set_light(idx, light_asset)

