"""
Pre-computation script: Generates cached 3D surface mesh JSON files for all BraTS benchmark cases.
Run this ONCE locally (or in Dockerfile) so Render serves meshes from disk instead of computing
Marching Cubes on every request (which causes 502s on the 512MB free tier).

Usage:
    python precompute_meshes.py

Output:
    data/segmentation/meshes_<case_key>.json  for each case
"""

import json
import sys
from pathlib import Path

# Make sure imports resolve regardless of CWD
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.segmentation.pipeline import load_nifti_file
from backend.segmentation.surface_mesh import generate_subregion_meshes

MANIFEST_PATH = ROOT / "data" / "segmentation" / "dataset_manifest.json"
DATA_DIR = ROOT / "data" / "segmentation"

def main():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    for case_key, case_info in manifest.items():
        out_path = DATA_DIR / f"meshes_{case_key}.json"
        if out_path.exists():
            print(f"[SKIP] {case_key} — mesh cache already exists at {out_path.name}")
            continue

        gt_filename = case_info["gt_path"].replace("\\", "/").split("/")[-1]
        gt_path = DATA_DIR / gt_filename

        if not gt_path.exists():
            print(f"[WARN] {case_key} — ground truth file not found: {gt_path}")
            continue

        print(f"[INFO] Processing {case_key} ({case_info['title']}) …", flush=True)
        try:
            gt_vol = load_nifti_file(gt_path)
            meshes = generate_subregion_meshes(gt_vol.data, gt_vol.voxel_spacing)
            with open(out_path, "w") as f:
                json.dump(meshes, f, separators=(",", ":"))
            verts = meshes.get("WT", {}) or {}
            vc = verts.get("vertex_count", 0)
            print(f"  ✓ Saved {out_path.name}  (WT vertices: {vc:,})")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n[DONE] All mesh caches written to data/segmentation/")

if __name__ == "__main__":
    main()
