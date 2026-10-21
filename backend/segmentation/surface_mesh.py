"""
3D Surface Mesh Generator for Volumetric MRI Segmentations
Extracts true watertight isosurfaces via Marching Cubes, formatted for Three.js WebGL rendering.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def extract_surface_mesh_from_mask(
    mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Optional[Dict[str, Any]]:
    """
    Extracts a true 3D surface mesh from a binary 3D segmentation mask using Marching Cubes.
    Returns None if the mask has insufficient foreground voxels (< 10).
    """
    binary_mask = (mask_3d > 0).astype(np.uint8)
    voxel_count = int(np.sum(binary_mask))
    if voxel_count < 10 or not SKIMAGE_AVAILABLE:
        return None

    dx, dy, dz = voxel_spacing

    try:
        # Adaptive step size to control triangle counts for smooth 60fps WebGL rendering and low memory footprint
        step = 2 if voxel_count > 1000 else 1
        if voxel_count > 100000:
            step = 3

        # Pad volume by 1 voxel to guarantee watertight isosurfaces at boundaries
        padded = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
        verts, faces, _, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=(dx, dy, dz),
            step_size=step
        )

        if len(verts) == 0 or len(faces) == 0:
            return None

        verts = np.asarray(verts, dtype=np.float32)
        verts[:, 0] -= dx
        verts[:, 1] -= dy
        verts[:, 2] -= dz

        # Center mesh at geometric centroid for intuitive orbital rotation
        centroid = np.mean(verts, axis=0)
        verts_centered = verts - centroid

        bbox = [
            round(float(np.max(verts[:, 0]) - np.min(verts[:, 0])), 2),
            round(float(np.max(verts[:, 1]) - np.min(verts[:, 1])), 2),
            round(float(np.max(verts[:, 2]) - np.min(verts[:, 2])), 2)
        ]

        return {
            "vertices": np.round(verts_centered.flatten(), 1).tolist(),
            "faces": faces.flatten().tolist(),
            "vertex_count": len(verts),
            "face_count": len(faces),
            "centroid": np.round(centroid, 2).tolist(),
            "bbox_mm": bbox,
            "voxel_count": voxel_count,
            "algorithm": "marching_cubes"
        }
    except Exception as e:
        print(f"[WARNING] Marching cubes extraction failed: {e}")
        return None


def generate_subregion_meshes(
    mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Any]:
    """
    Extracts individual 3D surface meshes for Whole Tumor (WT),
    Tumor Core (TC), and Enhancing Tumor (ET) compartments.
    """
    clean_mask = np.where(mask_3d == 4, 3, mask_3d)

    # 1. Whole Tumor (WT): Labels 1, 2, 3
    wt_mask = np.isin(clean_mask, [1, 2, 3]).astype(np.uint8)
    wt_mesh = extract_surface_mesh_from_mask(wt_mask, voxel_spacing)

    # 2. Tumor Core (TC): Labels 1, 3
    tc_mask = np.isin(clean_mask, [1, 3]).astype(np.uint8)
    tc_mesh = extract_surface_mesh_from_mask(tc_mask, voxel_spacing)

    # 3. Enhancing Tumor (ET): Label 3
    et_mask = (clean_mask == 3).astype(np.uint8)
    et_mesh = extract_surface_mesh_from_mask(et_mask, voxel_spacing)

    return {
        "WT": wt_mesh,
        "TC": tc_mesh,
        "ET": et_mesh,
        "has_mesh": any([wt_mesh is not None, tc_mesh is not None, et_mesh is not None])
    }
