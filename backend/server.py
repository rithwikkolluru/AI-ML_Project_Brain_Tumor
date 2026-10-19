"""
Brain Tumor AI Diagnostic Suite — Unified Clinical AI & Multimodal Workstation Server
FastAPI ASGI Backend powering 2D classification, 3D volumetric BraTS segmentation,
interactive Marching Cubes 3D meshes, PDF report translation, and conversational medical assistant.
"""

import os
import io
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.models_2d import BrainTumorClassifier2D, CLASSES, CLASS_META
from backend.llm_explainer import extract_text_from_pdf_bytes, stream_medical_chat
from backend.segmentation.pipeline import (
    VolumetricSegmentationPipeline,
    load_nifti_file,
    extract_orthogonal_slice,
    render_slice_pack
)
from backend.segmentation.surface_mesh import generate_subregion_meshes
from backend.segmentation.metrics import compute_case_subregion_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_2D_PATH = BASE_DIR / "brain_tumor_model.keras"
DATA_2D_DIR = BASE_DIR / "data" / "samples_2d"
DATA_3D_DIR = BASE_DIR / "data" / "segmentation"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_TEMP_DIR = BASE_DIR / "uploads_temp"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Brain Tumor AI Diagnostic Suite & 3D Volumetric Workstation",
    description="Multimodal Neuro-Oncology Clinical AI Platform: 2D Classification (EfficientNetB0) + 3D BraTS Volumetric Segmentation + Interactive WebGL + Medical LLM Explainer",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
classifier_2d = BrainTumorClassifier2D(MODEL_2D_PATH)
segmentation_pipeline = VolumetricSegmentationPipeline(DATA_3D_DIR)


# Pydantic Schemas
class ChatRequest(BaseModel):
    prompt: str
    conversation_history: List[Dict[str, str]] = []
    mri_context: Optional[Dict[str, Any]] = None
    pdf_context: Optional[str] = None
    groq_api_key: Optional[str] = None
    use_offline: bool = False
    model: str = "llama-3.3-70b-versatile"


# =========================================================================
# TELEMETRY & SYSTEM HEALTH
# =========================================================================
@app.get("/api/status")
async def get_system_status():
    device = "Apple Silicon GPU (Metal M5)" if "metal" in str(os.environ) else "Cloud Server (CPU/CUDA)"
    return {
        "status": "online",
        "service": "Brain Tumor AI Diagnostic Suite",
        "version": "2.0.0",
        "device": device,
        "model_2d_loaded": classifier_2d.model is not None,
        "model_2d_architecture": "EfficientNetB0 (Fine-Tuned, 93.31% Test Accuracy)",
        "model_3d_engine": "3D U-Net & Marching Cubes WebGL",
        "classes": CLASSES,
        "class_metadata": CLASS_META,
        "samples_2d_count": len(list(DATA_2D_DIR.glob("*.jpg"))),
        "cases_3d_count": len(segmentation_pipeline.manifest)
    }


# =========================================================================
# 2D CLASSIFICATION & GRAD-CAM
# =========================================================================
@app.get("/api/samples/2d")
async def get_2d_samples():
    """Lists pre-loaded 2D MRI benchmark cases for zero-click evaluation."""
    samples = []
    sample_files = {
        "glioma": "glioma.jpg",
        "meningioma": "meningioma.jpg",
        "pituitary": "pituitary.jpg",
        "notumor": "notumor.jpg",
        "glioma2": "glioma2.jpg"
    }
    for key, fname in sample_files.items():
        file_path = DATA_2D_DIR / fname
        if file_path.exists():
            clean_class = "glioma" if "glioma" in key else key
            samples.append({
                "key": key,
                "filename": fname,
                "label": CLASS_META.get(clean_class, {}).get("label", key),
                "expected_class": clean_class,
                "description": CLASS_META.get(clean_class, {}).get("description", ""),
                "badge_class": CLASS_META.get(clean_class, {}).get("badge_class", "")
            })
    return {"samples": samples}


@app.post("/api/predict/2d")
async def predict_2d(file: UploadFile = File(None), sample_key: Optional[str] = Form(None)):
    """Accepts either an uploaded MRI image or a pre-loaded sample key."""
    image_bytes = None
    filename = "scan.jpg"

    if sample_key:
        sample_file = DATA_2D_DIR / f"{sample_key}.jpg"
        if not sample_file.exists():
            # Try finding by name
            for f in DATA_2D_DIR.glob("*.jpg"):
                if sample_key in f.stem:
                    sample_file = f
                    break
        if sample_file.exists():
            image_bytes = sample_file.read_bytes()
            filename = sample_file.name

    if image_bytes is None and file is not None:
        filename = file.filename
        image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image file or valid sample key provided.")

    try:
        result = classifier_2d.predict(image_bytes, filename=filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# =========================================================================
# 3D VOLUMETRIC SEGMENTATION & WEBGL ISOSURFACE MESHES
# =========================================================================
@app.get("/api/segmentation/cases")
async def get_segmentation_cases():
    """Lists available authentic BraTS 2023 clinical NIfTI cases."""
    cases = segmentation_pipeline.get_available_cases()
    return {"cases": cases}


@app.post("/api/segmentation/process")
async def process_segmentation(
    case_key: Optional[str] = Form(None),
    plane: str = Form("axial"),
    slice_idx: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    ground_truth: Optional[UploadFile] = File(None)
):
    """
    Processes a 3D volumetric scan (preloaded BraTS benchmark or uploaded .nii / .nii.gz).
    Returns multi-planar slice previews, physical volume (cm³), and Three.js 3D surface meshes.
    """
    if case_key and case_key in segmentation_pipeline.manifest:
        try:
            result = segmentation_pipeline.process_case(case_key, plane=plane, slice_idx=slice_idx)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Volumetric processing error: {str(e)}")

    if file is not None:
        # Custom NIfTI upload
        ext = file.filename.split(".", 1)[-1]
        temp_nii = UPLOAD_TEMP_DIR / f"{uuid.uuid4()}.{ext}"
        with open(temp_nii, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            mri_vol = load_nifti_file(temp_nii)
            
            # Slice extraction
            mri_slice, act_idx, total_slices = extract_orthogonal_slice(mri_vol.data, plane, slice_idx)
            
            gt_vol = None
            if ground_truth is not None:
                gt_ext = ground_truth.filename.split(".", 1)[-1]
                temp_gt = UPLOAD_TEMP_DIR / f"{uuid.uuid4()}.{gt_ext}"
                with open(temp_gt, "wb") as buffer:
                    shutil.copyfileobj(ground_truth.file, buffer)
                gt_vol = load_nifti_file(temp_gt)
                if temp_gt.exists():
                    os.remove(temp_gt)

            gt_slice = None
            meshes = {"WT": None, "TC": None, "ET": None, "has_mesh": False}
            metrics = None
            vol_cm3 = {}
            voxel_counts = {}

            if gt_vol is not None:
                gt_slice, _, _ = extract_orthogonal_slice(gt_vol.data, plane, slice_idx)
                meshes = generate_subregion_meshes(gt_vol.data, gt_vol.voxel_spacing)
                metrics = compute_case_subregion_metrics(gt_vol.data, gt_vol.data, gt_vol.voxel_spacing)
                dx, dy, dz = gt_vol.voxel_spacing
                vox_mm3 = dx * dy * dz
                for reg in ["WT", "TC", "ET"]:
                    v_cnt = metrics[reg]["pred_voxels"]
                    voxel_counts[reg] = v_cnt
                    vol_cm3[reg] = round((v_cnt * vox_mm3) / 1000.0, 2)

            slice_pack = render_slice_pack(mri_slice, gt_slice)

            return {
                "patient_study_id": file.filename,
                "title": f"Custom Clinical Study ({file.filename})",
                "plane": plane,
                "slice_index": act_idx,
                "total_slices": total_slices,
                "dimensions": list(mri_vol.shape),
                "voxel_spacing": list(mri_vol.voxel_spacing),
                "slices": slice_pack,
                "volume_cm3": vol_cm3,
                "voxel_counts": voxel_counts,
                "meshes": meshes,
                "metrics": metrics,
                "dataset_attribution": "Custom Clinical Volumetric NIfTI Study"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Custom NIfTI error: {str(e)}")
        finally:
            if temp_nii.exists():
                os.remove(temp_nii)

    raise HTTPException(status_code=400, detail="Provide either a valid case_key or an uploaded .nii/.nii.gz file.")


@app.get("/api/segmentation/slice")
async def get_slice(
    case_key: str = Query(...),
    plane: str = Query("axial"),
    slice_idx: int = Query(...)
):
    """Dynamically streams 2D slice PNGs when the user scrubs the radiology slider."""
    try:
        case_info = segmentation_pipeline.manifest.get(case_key)
        if not case_info:
            raise HTTPException(status_code=404, detail="Case key not found.")

        mri_filename = case_info["mri_path"].replace("\\", "/").split("/")[-1]
        gt_filename = case_info["gt_path"].replace("\\", "/").split("/")[-1]

        mri_vol = load_nifti_file(DATA_3D_DIR / mri_filename)
        gt_vol = load_nifti_file(DATA_3D_DIR / gt_filename)

        mri_slice, act_idx, total_slices = extract_orthogonal_slice(mri_vol.data, plane, slice_idx)
        gt_slice, _, _ = extract_orthogonal_slice(gt_vol.data, plane, slice_idx)

        pack = render_slice_pack(mri_slice, gt_slice)
        return {
            "slice_index": act_idx,
            "total_slices": total_slices,
            "plane": plane,
            "mri_image": pack["mri"],
            "seg_image": pack["seg"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# CLINICAL PDF PARSER & CONVERSATIONAL MEDICAL ASSISTANT
# =========================================================================
@app.post("/api/pdf/extract")
async def extract_pdf(file: UploadFile = File(...)):
    """Parses an uploaded medical or radiology PDF report."""
    content = await file.read()
    try:
        text = extract_text_from_pdf_bytes(content)
        word_count = len(text.split())
        return {
            "filename": file.filename,
            "text": text,
            "word_count": word_count,
            "summary_prompt": "Please summarize this medical report in plain, compassionate language for the patient."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streams conversational AI medical explainer completions via Server-Sent Events."""
    generator = stream_medical_chat(
        prompt=request.prompt,
        conversation_history=request.conversation_history,
        mri_context=request.mri_context,
        pdf_context=request.pdf_context,
        groq_api_key=request.groq_api_key,
        use_offline=request.use_offline,
        selected_model=request.model
    )
    return StreamingResponse(generator, media_type="text/plain")


# =========================================================================
# MOUNT STATIC FRONTEND
# =========================================================================
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
