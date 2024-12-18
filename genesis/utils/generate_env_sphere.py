import trimesh
import numpy as np
from PIL import Image


def compute_uv_from_vertex(vertex):
    """
    Convert vertex coordinates (x, y, z) to UV coordinates (u, v)
    for a spherical mapping.
    """
    vertex_new = np.array(vertex)
    r = np.linalg.norm(vertex)
    if r == 0:
        return 0, 0, vertex_new

    phi = np.arctan2(vertex[2], np.linalg.norm(vertex[0:2]))
    theta = np.arctan2(vertex[1], vertex[0])

    u = 0.5 - theta / (2 * np.pi)
    v = phi / np.pi - 0.5

    mod = theta / (np.pi / 32) % 1

    if (mod > 0.5 and mod < 0.9999) or (mod <= 0.5 and mod > 0.0001):
        # print(theta, theta/(np.pi/32)%1, vertex, np.cos(phi))
        # print(0.5 - theta / (2 * np.pi))
        vertex_new = np.array([-np.cos(phi), 0, vertex[2]])
        u = 1.0

    return u, v, vertex_new


def main():
    # Create a sphere mesh
    phi = np.linspace(0, np.pi * 2, 65)[:-1]
    phi = np.concatenate([phi[:33], [np.pi + 0.01], phi[33:]])

    sphere = trimesh.creation.uv_sphere(radius=1.0, phi=phi)
    faces = sphere.faces[:, ::-1]
    normals = -sphere.vertex_normals

    vertices = []
    uv_coords = []
    for vertex in np.array(sphere.vertices):
        u, v, vertex_new = compute_uv_from_vertex(vertex)
        uv_coords.append([u, v])
        vertices.append(vertex_new)

    uv_coords = np.vstack(uv_coords)
    vertices = np.vstack(vertices)
    sphere = trimesh.Trimesh(vertices, faces, normals, process=False)

    # Add UV coordinates to the sphere's visual attribute
    sphere.visual = trimesh.visual.TextureVisuals(
        uv=uv_coords, image=Image.open("/home/zhouxian/Downloads/env_maps/bathroom_01.jpg")
    )
    # Export the mesh with the texture mapped
    sphere.export("env_sphere.obj")


if __name__ == "__main__":
    main()
