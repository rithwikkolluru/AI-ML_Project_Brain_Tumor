"""
Unit tests for 3D Marching Cubes surface mesh generator
"""

import numpy as np
from backend.segmentation.surface_mesh import extract_surface_mesh_from_mask, generate_subregion_meshes


def test_surface_mesh_synthetic_sphere():
    # Create 30x30x30 volume with a solid sphere
    vol = np.zeros((30, 30, 30), dtype=np.uint8)
    for x in range(30):
        for y in range(30):
            for z in range(30):
                if (x - 15)**2 + (y - 15)**2 + (z - 15)**2 <= 8**2:
                    vol[x, y, z] = 1

    mesh = extract_surface_mesh_from_mask(vol, voxel_spacing=(1.0, 1.0, 1.0))
    assert mesh is not None
    assert mesh["vertex_count"] > 100
    assert mesh["face_count"] > 100
    assert len(mesh["vertices"]) == mesh["vertex_count"] * 3
    assert len(mesh["faces"]) == mesh["face_count"] * 3
    assert "bbox_mm" in mesh
    assert mesh["algorithm"] == "marching_cubes"


def test_surface_mesh_empty_mask():
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    mesh = extract_surface_mesh_from_mask(vol)
    assert mesh is None


def test_subregion_meshes():
    vol = np.zeros((30, 30, 30), dtype=np.uint8)
    # Label 1 (Necrotic), Label 2 (Edema), Label 3 (Enhancing)
    vol[10:20, 10:20, 10:20] = 2
    vol[12:18, 12:18, 12:18] = 1
    vol[13:17, 13:17, 13:17] = 3

    meshes = generate_subregion_meshes(vol)
    assert meshes["has_mesh"] is True
    assert meshes["WT"] is not None
    assert meshes["TC"] is not None
    assert meshes["ET"] is not None
