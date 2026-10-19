# 🧠 Brain Tumor AI Diagnostic Suite & 3D Volumetric Workstation

An end-to-end Multimodal Medical AI platform combining high-accuracy Brain MRI Deep Learning classification (EfficientNetB0, 93.31% test accuracy, 100% normal recall), Grad-CAM explainability, 3D Volumetric BraTS Tumor Segmentation with interactive Three.js WebGL isosurfaces, and a Conversational AI Medical Assistant for clinical report translation.

---

## 🌟 Key Highlights & Architectural Advantages

- **🔬 State-of-the-Art 2D Vision Backbone**: Fine-tuned EfficientNetB0 on 7,023 MRI scans with two-stage transfer learning.
  - **93.31% Overall Test Accuracy** on 1,600 unseen clinical test scans.
  - **100.00% Recall on Normal Brain Scans** (Zero false positive tumor detections).
  - **100.00% Recall on Pituitary Sellar Lesions**.
- **🌡️ Vectorized Grad-CAM Saliency Maps**: Real-time convolutional feature activation maps with dynamic opacity blending (0–100%) and layer toggle modes (Raw, Heatmap, Superimposed).
- **🧊 3D Volumetric Segmentation Lab (BraTS Protocol)**:
  - Multi-Planar Reconstruction (MPR) across synchronized **Axial (Z)**, **Coronal (Y)**, and **Sagittal (X)** orthogonal planes.
  - High-performance slice scrubbing with 18 fps cine playback.
  - BraTS subcompartment deconstruction: **Whole Tumor (WT)**, **Tumor Core (TC)**, **Enhancing Margin (ET)**, and **Peritumoral Edema (ED)**.
  - Integration of physical voxel spacing ($dx \times dy \times dz$) for exact anatomical volume quantification in $\text{cm}^3$ and $\text{mm}^3$.
- **🌐 Interactive 3D WebGL Isosurface Meshes**: Marching Cubes surface extraction rendered live in Three.js with orbital rotation, zoom, auto-rotate, and subregion layer toggles.
- **💬 Multimodal AI Medical Assistant & PDF Explainer**: Integrated conversational doctor assistant (Groq Llama 3.3 / Llama 3.1 or Local Ollama Gemma 4) that converts dense radiology reports into patient-friendly explanations and suggested clinical questions.
- **☁️ 1-Click Render Cloud Deployment**: Fully containerized with production `Dockerfile`, cross-platform Linux compatibility, memory optimization, and `render.yaml` blueprint.

---

## 📊 Benchmark Test Performance (1,600 Unseen Scans)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | **97.28%** | 80.50% | 88.10% | 400 |
| **Meningioma** | **88.54%** | **92.75%** | **90.60%** | 400 |
| **No Tumor (Control)** | **93.02%** | **100.00%** | **96.39%** | 400 |
| **Pituitary Adenoma** | **95.24%** | **100.00%** | **97.56%** | 400 |
| **Overall Accuracy** | — | — | **93.31%** | **1,600** |
| **Macro Average** | **93.52%** | **93.31%** | **93.16%** | **1,600** |

---

## 🏗️ System Architecture

```text
                                  ┌───────────────────────────┐
                                  │   Multi-Modal Inputs      │
                                  │ 2D JPG/PNG • 3D NIfTI • PDF│
                                  └─────────────┬─────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
     ┌─────────────────────────────┐                       ┌─────────────────────────────┐
     │ 2D Vision Pipeline          │                       │ 3D Volumetric Pipeline      │
     │ • EfficientNetB0 (93.31%)   │                       │ • BraTS NIfTI Parser        │
     │ • Grad-CAM Feature Maps     │                       │ • Multi-Planar MPR Slices   │
     │ • Class Probabilities       │                       │ • Marching Cubes 3D Mesh    │
     └──────────────┬──────────────┘                       └──────────────┬──────────────┘
                    │                                                     │
                    └──────────────────────────┬──────────────────────────┘
                                               ▼
                               ┌─────────────────────────────┐
                               │ Multimodal LLM Explainer    │
                               │ • PyPDF Clinical Extractor  │
                               │ • Groq Cloud / Local Ollama │
                               │ • Patient Medical Q&A       │
                               └──────────────┬──────────────┘
                                               ▼
                               ┌─────────────────────────────┐
                               │ PACS Clinical Workstation   │
                               │ Dark Canvas • Three.js WebGL│
                               │ Cine 18fps • Printable Rep. │
                               └─────────────────────────────┘
```

---

## 🚀 Running Locally

### 1. Setup Environment
```bash
conda activate braintumor
pip install -r requirements-render.txt
```

### 2. Run Test Suite
```bash
pytest -v tests/
```

### 3. Start Unified Clinical Workstation
```bash
python main.py
```
Open **`http://localhost:8000`** in your browser.

*(Optional)* Run legacy Streamlit interface:
```bash
streamlit run app.py
```

---

## ☁️ 1-Click Deployment to Render

This repository includes a production `Dockerfile` and `render.yaml` pre-configured for Render Web Services.

### Steps to Deploy on Render:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Complete clinical workstation with 3D WebGL and Render setup"
   git push origin feature/clinical-workstation-3d-render
   ```
2. **Open Render Dashboard** ([render.com](https://render.com)):
   - Click **New +** $\longrightarrow$ **Blueprint** (or **Web Service**).
   - Connect your GitHub repository.
   - If using **Blueprint**, Render will detect `render.yaml` automatically and configure:
     - Runtime: **Docker**
     - Health Check Path: `/api/status`
     - Port: Auto-bound via `$PORT`
3. **Environment Variables (Optional)**:
   - `GROQ_API_KEY`: *(Optional)* Add your free Groq API key in the Render Environment tab to enable cloud AI conversational responses.

---

## ⚠️ Clinical Research Disclaimer
**FOR INVESTIGATIONAL RESEARCH AND DEMONSTRATION USE ONLY.**  
This platform is an artificial intelligence research workstation developed for educational evaluation and surgical planning investigation. It has **NOT** been cleared or certified by the FDA, EMA, or any healthcare regulatory body. It must **never** be used for primary diagnosis or direct clinical patient management.
