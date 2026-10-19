"""
Supervised Validation Metrics & Volumetric Morphometrics
Computes Dice Similarity Coefficient (DSC), IoU (Jaccard Index), Sensitivity,
and true physical volume integration (cm³ and mm³) using voxel spacing (dx, dy, dz).
"""

from typing import Dict, Any, Tuple
import numpy as np


def compute_binary_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray) -> Dict[str, float]:
    """
    Computes rigorous DSC, IoU, and Sensitivity between binary masks.
    Handles empty-mask true-negatives (Dice=1.0 if both empty).
    """
    p = (pred_binary > 0).astype(np.bool_)
    g = (gt_binary > 0).astype(np.bool_)

    intersection = int(np.logical_and(p, g).sum())
    p_sum = int(p.sum())
    g_sum = int(g.sum())
    union = int(np.logical_or(p, g).sum())

    # Empty mask handling
    if p_sum == 0 and g_sum == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "sensitivity": 1.0,
            "pred_voxels": 0,
            "gt_voxels": 0
        }

    if p_sum == 0 or g_sum == 0:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "sensitivity": 0.0,
            "pred_voxels": p_sum,
            "gt_voxels": g_sum
        }

    dice = (2.0 * intersection) / (p_sum + g_sum)
    iou = intersection / union if union > 0 else 0.0
    sensitivity = intersection / g_sum if g_sum > 0 else 0.0

    return {
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "sensitivity": round(float(sensitivity), 4),
        "pred_voxels": p_sum,
        "gt_voxels": g_sum
    }


def compute_case_subregion_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Any]:
    """
    Calculates BraTS benchmark subregion metrics for:
      - Whole Tumor (WT): Labels 1, 2, 3
      - Tumor Core (TC): Labels 1, 3
      - Enhancing Tumor (ET): Label 3
    """
    p_clean = np.where(pred_mask == 4, 3, pred_mask)
    g_clean = np.where(gt_mask == 4, 3, gt_mask)

    # Whole Tumor (WT)
    p_wt = np.isin(p_clean, [1, 2, 3])
    g_wt = np.isin(g_clean, [1, 2, 3])
    wt_metrics = compute_binary_metrics(p_wt, g_wt)

    # Tumor Core (TC)
    p_tc = np.isin(p_clean, [1, 3])
    g_tc = np.isin(g_clean, [1, 3])
    tc_metrics = compute_binary_metrics(p_tc, g_tc)

    # Enhancing Tumor (ET)
    p_et = (p_clean == 3)
    g_et = (g_clean == 3)
    et_metrics = compute_binary_metrics(p_et, g_et)

    # Voxel volume factor in mm³
    dx, dy, dz = voxel_spacing
    voxel_vol_mm3 = dx * dy * dz

    return {
        "WT": {
            **wt_metrics,
            "pred_vol_cm3": round((wt_metrics["pred_voxels"] * voxel_vol_mm3) / 1000.0, 2),
            "gt_vol_cm3": round((wt_metrics["gt_voxels"] * voxel_vol_mm3) / 1000.0, 2)
        },
        "TC": {
            **tc_metrics,
            "pred_vol_cm3": round((tc_metrics["pred_voxels"] * voxel_vol_mm3) / 1000.0, 2),
            "gt_vol_cm3": round((tc_metrics["gt_voxels"] * voxel_vol_mm3) / 1000.0, 2)
        },
        "ET": {
            **et_metrics,
            "pred_vol_cm3": round((et_metrics["pred_voxels"] * voxel_vol_mm3) / 1000.0, 2),
            "gt_vol_cm3": round((et_metrics["gt_voxels"] * voxel_vol_mm3) / 1000.0, 2)
        },
        "mean_dice": round(float(np.mean([wt_metrics["dice"], tc_metrics["dice"], et_metrics["dice"]])), 4)
    }
