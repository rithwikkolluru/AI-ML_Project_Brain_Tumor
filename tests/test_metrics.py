"""
Unit tests for validation metrics (Dice, IoU, Sensitivity, Volume)
"""

import numpy as np
from backend.segmentation.metrics import compute_binary_metrics, compute_case_subregion_metrics


def test_perfect_dice():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1

    res = compute_binary_metrics(mask, mask)
    assert res["dice"] == 1.0
    assert res["iou"] == 1.0
    assert res["sensitivity"] == 1.0


def test_empty_both_masks():
    mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask2 = np.zeros((10, 10, 10), dtype=np.uint8)

    res = compute_binary_metrics(mask1, mask2)
    assert res["dice"] == 1.0
    assert res["iou"] == 1.0


def test_disjoint_masks():
    m1 = np.zeros((10, 10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10, 10), dtype=np.uint8)
    m1[0:3, 0:3, 0:3] = 1
    m2[6:9, 6:9, 6:9] = 1

    res = compute_binary_metrics(m1, m2)
    assert res["dice"] == 0.0
    assert res["iou"] == 0.0


def test_subregion_metrics():
    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    gt[5:15, 5:15, 5:15] = 2  # Edema
    gt[7:13, 7:13, 7:13] = 1  # Core
    gt[8:12, 8:12, 8:12] = 3  # Enhancing

    sub_res = compute_case_subregion_metrics(gt, gt, voxel_spacing=(1.0, 1.0, 1.0))
    assert sub_res["WT"]["dice"] == 1.0
    assert sub_res["TC"]["dice"] == 1.0
    assert sub_res["ET"]["dice"] == 1.0
    assert sub_res["WT"]["pred_vol_cm3"] > 0
