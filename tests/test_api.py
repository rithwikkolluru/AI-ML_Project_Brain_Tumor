"""
Integration tests for FastAPI endpoints
"""

from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)


def test_api_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["model_2d_loaded"] is True
    assert "EfficientNetB0" in data["model_2d_architecture"]


def test_api_2d_samples():
    res = client.get("/api/samples/2d")
    assert res.status_code == 200
    data = res.json()
    assert len(data["samples"]) >= 4


def test_api_2d_predict_sample():
    res = client.post("/api/predict/2d", data={"sample_key": "glioma"})
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert "confidence" in data
    assert "gradcam_heatmap" in data
    assert "probabilities" in data


def test_api_segmentation_cases():
    res = client.get("/api/segmentation/cases")
    assert res.status_code == 200
    data = res.json()
    assert len(data["cases"]) == 6


def test_api_segmentation_process_case_001():
    res = client.post("/api/segmentation/process", data={"case_key": "case_001", "plane": "axial", "slice_idx": 71})
    assert res.status_code == 200
    data = res.json()
    assert data["patient_study_id"] == "BraTS-GLI-00000-000"
    assert "volume_cm3" in data
    assert data["volume_cm3"]["WT"] > 0
    assert data["meshes"]["has_mesh"] is True


def test_api_segmentation_slice():
    res = client.get("/api/segmentation/slice?case_key=case_001&plane=axial&slice_idx=71")
    assert res.status_code == 200
    data = res.json()
    assert "mri_image" in data
    assert "seg_image" in data
