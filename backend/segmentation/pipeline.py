"""
3D Volumetric NIfTI Pipeline & Multi-Planar Slice Engine
Loads volumetric MRI (.nii / .nii.gz), performs non-zero normalization,
extracts synchronized Axial, Coronal, and Sagittal orthogonal slices,
and integrates voxel spacing for true physical measurements.
"""

import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
from PIL import Image

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

from backend.segmentation.model import LABEL_DEFINITIONS, SUBREGION_HIERARCHY
from backend.segmentation.surface_mesh import generate_subregion_meshes
from backend.segmentation.metrics import compute_case_subregion_metrics

# Color LUT for BraTS overlay rendering [R, G, B, A]
LABEL_RGBA_LUT = np.array([
    [0, 0, 0, 0],          # 0: BG
    [255, 59, 48, 190],    # 1: NCR / Crimson
    [57, 213, 255, 170],   # 2: ED / Cyan-blue
    [255, 214, 10, 220]    # 3: ET / Gold
], dtype=np.uint8)


class NiftiVolume:
    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        shape: Tuple[int, ...]
    ):
        self.data = data
        self.affine = affine
        self.voxel_spacing = voxel_spacing  # (dx, dy, dz) mm
        self.shape = shape


def load_nifti_file(file_path: Union[str, Path]) -> NiftiVolume:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    if not NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel is required to load NIfTI volumes.")

    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    raw_data = np.asarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)

    try:
        zooms = img.header.get_zooms()
        dx = float(zooms[0]) if len(zooms) > 0 else 1.0
        dy = float(zooms[1]) if len(zooms) > 1 else 1.0
        dz = float(zooms[2]) if len(zooms) > 2 else 1.0
    except Exception:
        dx = float(np.sqrt(np.sum(affine[:3, 0] ** 2)))
        dy = float(np.sqrt(np.sum(affine[:3, 1] ** 2)))
        dz = float(np.sqrt(np.sum(affine[:3, 2] ** 2)))

    voxel_spacing = (round(dx, 3), round(dy, 3), round(dz, 3))

    if raw_data.ndim == 4:
        # Extract first channel if 4D
        raw_data = raw_data[..., 0]

    return NiftiVolume(
        data=raw_data,
        affine=affine,
        voxel_spacing=voxel_spacing,
        shape=raw_data.shape
    )


def normalize_mri_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalizes 2D slice intensities to 0-255 uint8 grayscale."""
    non_zero = slice_2d[slice_2d > 0]
    if len(non_zero) == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    p1 = np.percentile(non_zero, 1)
    p99 = np.percentile(non_zero, 99)
    if p99 > p1:
        clipped = np.clip(slice_2d, p1, p99)
        norm = ((clipped - p1) / (p99 - p1)) * 255.0
        return norm.astype(np.uint8)
    return np.zeros_like(slice_2d, dtype=np.uint8)


def extract_orthogonal_slice(
    volume_data: np.ndarray,
    plane: str = "axial",
    slice_idx: Optional[int] = None
) -> Tuple[np.ndarray, int, int]:
    """
    Extracts a 2D slice from 3D matrix (H, W, D) across:
      - axial: Z plane
      - coronal: Y plane
      - sagittal: X plane
    """
    h, w, d = volume_data.shape
    if plane == "axial":
        max_idx = d - 1
        idx = max(0, min(slice_idx if slice_idx is not None else d // 2, max_idx))
        slice_2d = volume_data[:, :, idx]
        # Orient for standard PACS display
        slice_2d = np.rot90(slice_2d, 1)
    elif plane == "coronal":
        max_idx = w - 1
        idx = max(0, min(slice_idx if slice_idx is not None else w // 2, max_idx))
        slice_2d = volume_data[:, idx, :]
        slice_2d = np.rot90(slice_2d, 1)
    elif plane == "sagittal":
        max_idx = h - 1
        idx = max(0, min(slice_idx if slice_idx is not None else h // 2, max_idx))
        slice_2d = volume_data[idx, :, :]
        slice_2d = np.rot90(slice_2d, 1)
    else:
        raise ValueError(f"Unknown anatomical plane: {plane}")

    return slice_2d, idx, max_idx + 1


def render_slice_pack(
    mri_slice: np.ndarray,
    seg_slice: Optional[np.ndarray] = None
) -> Dict[str, str]:
    """Renders raw MRI grayscale PNG and segmentation RGBA PNG as Base64."""
    mri_uint8 = normalize_mri_slice(mri_slice)
    img_mri = Image.fromarray(mri_uint8, mode="L").convert("RGB")
    buf_mri = io.BytesIO()
    img_mri.save(buf_mri, format="PNG")
    mri_b64 = "data:image/png;base64," + base64.b64encode(buf_mri.getvalue()).decode("utf-8")

    seg_b64 = None
    if seg_slice is not None:
        clean_seg = np.where(seg_slice == 4, 3, seg_slice).astype(np.int64)
        clean_seg = np.clip(clean_seg, 0, 3)
        rgba = LABEL_RGBA_LUT[clean_seg]
        img_seg = Image.fromarray(rgba, mode="RGBA")
        buf_seg = io.BytesIO()
        img_seg.save(buf_seg, format="PNG")
        seg_b64 = "data:image/png;base64," + base64.b64encode(buf_seg.getvalue()).decode("utf-8")

    return {
        "mri": mri_b64,
        "seg": seg_b64
    }


class VolumetricSegmentationPipeline:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        manifest_file = self.data_dir / "dataset_manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not load dataset_manifest.json: {e}")
        return {}

    def get_available_cases(self) -> List[Dict[str, Any]]:
        cases = []
        for key, val in self.manifest.items():
            cases.append({
                "key": key,
                "id": val.get("id", key),
                "title": val.get("title", key),
                "description": val.get("description", ""),
                "volume_cm3": val.get("volume_cm3", {}),
                "voxel_counts": val.get("voxel_counts", {}),
                "initial_slice": val.get("initial_slice_focus", 70)
            })
        return cases

    def process_case(
        self,
        case_key: str,
        plane: str = "axial",
        slice_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        case_info = self.manifest.get(case_key)
        if not case_info:
            raise KeyError(f"Case '{case_key}' not found in manifest.")

        mri_filename = case_info["mri_path"].replace("\\", "/").split("/")[-1]
        gt_filename = case_info["gt_path"].replace("\\", "/").split("/")[-1]

        mri_path = self.data_dir / mri_filename
        gt_path = self.data_dir / gt_filename

        mri_vol = load_nifti_file(mri_path)
        gt_vol = load_nifti_file(gt_path)

        # Slice extraction
        target_slice = slice_idx if slice_idx is not None else case_info.get("initial_slice_focus", 70)
        mri_slice, act_idx, total_slices = extract_orthogonal_slice(mri_vol.data, plane, target_slice)
        gt_slice, _, _ = extract_orthogonal_slice(gt_vol.data, plane, target_slice)

        slice_pack = render_slice_pack(mri_slice, gt_slice)

        # Generate or retrieve cached 3D Meshes for Three.js (prevents CPU/RAM spikes)
        if not hasattr(self, "_mesh_cache"):
            self._mesh_cache = {}

        if case_key not in self._mesh_cache:
            self._mesh_cache[case_key] = generate_subregion_meshes(gt_vol.data, gt_vol.voxel_spacing)
        meshes = self._mesh_cache[case_key]

        # Volumetric metrics
        metrics = compute_case_subregion_metrics(gt_vol.data, gt_vol.data, gt_vol.voxel_spacing)

        return {
            "case_key": case_key,
            "patient_study_id": case_info.get("id"),
            "title": case_info.get("title"),
            "plane": plane,
            "slice_index": act_idx,
            "total_slices": total_slices,
            "dimensions": list(mri_vol.shape),
            "voxel_spacing": list(mri_vol.voxel_spacing),
            "slices": slice_pack,
            "volume_cm3": case_info.get("volume_cm3", {}),
            "voxel_counts": case_info.get("voxel_counts", {}),
            "meshes": meshes,
            "metrics": metrics,
            "dataset_attribution": "ASNR-MICCAI BraTS 2023 Adult Glioma Challenge (CC BY-SA 4.0)"
        }
