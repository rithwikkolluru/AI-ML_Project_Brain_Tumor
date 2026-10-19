/**
 * Brain Tumor AI Diagnostic Suite — Clinical Workstation Controller
 * 2D Classifier Home, Drag-and-Drop MRI Ingestion, 3D Marching Cubes WebGL, MPR Slicing & Theme Engine.
 */

(function () {
  "use strict";

  const API_BASE = window.location.origin;

  // Active Application State
  const state = {
    activeTab: "scan2d",
    currentTheme: localStorage.getItem("bt_theme") || "dark",
    scan2d: null,
    stagedFile: null,
    scan3d: null,
    current3DCaseKey: "case_001",
    currentPlane: "axial",
    currentSliceIdx: 71,
    cineInterval: null,
    isCinePlaying: false,
    autoRotate: true,
    threeScene: null,
    threeCamera: null,
    threeRenderer: null,
    threeControls: null,
    threeMeshes: {},
    chatHistory: [],
    pdfText: null
  };

  // =========================================================================
  // INITIALIZATION & TAB SWITCHING
  // =========================================================================
  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initTabs();          // Three.js is lazy-initialised on first 3D tab click
    fetchSystemStatus();
    loadPresets();
    setupDropzone();
    setup2DControls();
    setup3DControls();
    setupChatControls();
  });

  // =========================================================================
  // LIGHT / DARK THEME CONTROLLER
  // =========================================================================
  function initTheme() {
    applyTheme(state.currentTheme);

    const themeBtn = document.getElementById("themeToggleBtn");
    if (themeBtn) {
      themeBtn.addEventListener("click", () => {
        const nextTheme = state.currentTheme === "dark" ? "light" : "dark";
        state.currentTheme = nextTheme;
        localStorage.setItem("bt_theme", nextTheme);
        applyTheme(nextTheme);
      });
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const sunIcon = document.getElementById("themeIconSun");
    const moonIcon = document.getElementById("themeIconMoon");
    const toggleText = document.getElementById("themeToggleText");

    if (theme === "light") {
      if (sunIcon) sunIcon.style.display = "inline-block";
      if (moonIcon) moonIcon.style.display = "none";
      if (toggleText) toggleText.textContent = "Dark Mode";
      if (state.threeScene && typeof THREE !== "undefined") state.threeScene.background = new THREE.Color(0xf1f5f9);
    } else {
      if (sunIcon) sunIcon.style.display = "none";
      if (moonIcon) moonIcon.style.display = "inline-block";
      if (toggleText) toggleText.textContent = "Light Mode";
      if (state.threeScene && typeof THREE !== "undefined") state.threeScene.background = new THREE.Color(0x050811);
    }
  }

  function initTabs() {
    const tabBtns = document.querySelectorAll(".nav-tab-btn");
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        switchTab(targetTab);
      });
    });

    const homeBtn = document.getElementById("brandHomeBtn");
    if (homeBtn) homeBtn.addEventListener("click", () => switchTab("scan2d"));
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    document.querySelectorAll(".nav-tab-btn").forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-pane").forEach(pane => {
      pane.classList.toggle("active", pane.id === `tab-${tabId}`);
    });

    if (tabId === "segmentation3d") {
      // Lazy-init Three.js now that the container is visible and has real dimensions
      ensureThreeJs();
      setTimeout(() => onWindowResize(), 80);
    }
  }

  // =========================================================================
  // TELEMETRY & PRESETS
  // =========================================================================
  async function fetchSystemStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) {
        const data = await res.json();
        const devElem = document.getElementById("telemetryDevice");
        if (devElem) devElem.textContent = data.device.toUpperCase();
      }
    } catch (err) {
      console.warn("Status fetch failed:", err);
    }
  }

  async function loadPresets() {
    // 1. Load 2D Presets into Home Page Preset Grid
    try {
      const res = await fetch(`${API_BASE}/api/samples/2d`);
      if (res.ok) {
        const data = await res.json();
        render2DPresets(data.samples);
      }
    } catch (err) {
      console.warn("Failed to load 2D presets:", err);
    }

    // 2. Load 3D Cases
    try {
      const res = await fetch(`${API_BASE}/api/segmentation/cases`);
      if (res.ok) {
        const data = await res.json();
        render3DCases(data.cases);
        loadSegmentationCase("case_001");
      }
    } catch (err) {
      console.warn("Failed to load 3D cases:", err);
    }
  }

  function render2DPresets(samples) {
    const container = document.getElementById("presetPillsGrid");
    if (!container) return;
    container.innerHTML = "";

    samples.forEach(sample => {
      const pill = document.createElement("div");
      pill.className = "preset-pill";
      pill.innerHTML = `
        <img class="preset-pill-img" src="/assets/samples/${sample.filename}" alt="${sample.label}">
        <div>
          <div class="preset-pill-name">${sample.label}</div>
          <span class="badge ${sample.badge_class}" style="font-size: 0.68rem;">${sample.expected_class.toUpperCase()}</span>
        </div>
      `;
      pill.addEventListener("click", () => {
        analyze2DSample(sample.key);
      });
      container.appendChild(pill);
    });
  }

  function render3DCases(cases) {
    const sidebarContainer = document.getElementById("segCasesSelector");
    if (!sidebarContainer) return;
    sidebarContainer.innerHTML = "";

    cases.forEach(c => {
      const btn = document.createElement("button");
      btn.className = `btn btn-secondary btn-block ${c.key === state.current3DCaseKey ? "btn-primary" : ""}`;
      btn.style.width = "100%";
      btn.style.textAlign = "left";
      btn.style.justifyContent = "flex-start";
      btn.innerHTML = `<strong>${c.title}</strong><br><span style="font-size: 0.72rem; opacity: 0.75;">${c.id} &bull; WT: ${c.volume_cm3?.WT || "—"} cm³</span>`;
      btn.addEventListener("click", () => {
        loadSegmentationCase(c.key);
      });
      sidebarContainer.appendChild(btn);
    });
  }

  // =========================================================================
  // DRAG-AND-DROP MRI UPLOAD SECTION
  // =========================================================================
  function setupDropzone() {
    const dropzone = document.getElementById("mriDropzone");
    const fileInput = document.getElementById("inputUpload2D");
    const browseBtn = document.getElementById("btnBrowseMri");
    const stageCard = document.getElementById("fileStageCard");
    const stageThumbnail = document.getElementById("stageThumbnail");
    const stageFilename = document.getElementById("stageFilename");
    const stageMeta = document.getElementById("stageMeta");
    const btnCancelStage = document.getElementById("btnCancelStage");
    const btnExecuteAnalysis = document.getElementById("btnExecuteAnalysis");

    if (!dropzone || !fileInput) return;

    // Trigger browse
    dropzone.addEventListener("click", (e) => {
      if (e.target !== browseBtn) fileInput.click();
    });
    if (browseBtn) {
      browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
      });
    }

    // Drag-and-drop events
    ["dragenter", "dragover"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        handleSelectedFile(files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleSelectedFile(e.target.files[0]);
      }
    });

    function handleSelectedFile(file) {
      if (!file.type.startsWith("image/")) {
        alert("Please select a valid MRI image file (.jpg, .jpeg, .png)");
        return;
      }

      state.stagedFile = file;
      const sizeKb = (file.size / 1024).toFixed(1);

      const reader = new FileReader();
      reader.onload = (ev) => {
        if (stageThumbnail) stageThumbnail.src = ev.target.result;
        if (stageFilename) stageFilename.textContent = file.name;
        if (stageMeta) stageMeta.textContent = `${sizeKb} KB • Ready for EfficientNetB0 classification`;
        if (stageCard) stageCard.style.display = "block";
      };
      reader.readAsDataURL(file);
    }

    if (btnCancelStage) {
      btnCancelStage.addEventListener("click", () => {
        state.stagedFile = null;
        fileInput.value = "";
        if (stageCard) stageCard.style.display = "none";
      });
    }

    if (btnExecuteAnalysis) {
      btnExecuteAnalysis.addEventListener("click", async () => {
        if (!state.stagedFile) return;
        const formData = new FormData();
        formData.append("file", state.stagedFile);
        btnExecuteAnalysis.textContent = "Analyzing Scan...";
        btnExecuteAnalysis.disabled = true;

        try {
          await analyze2DScan(formData);
        } finally {
          btnExecuteAnalysis.textContent = "Run Deep Learning Analysis";
          btnExecuteAnalysis.disabled = false;
        }
      });
    }
  }

  // =========================================================================
  // 2D CLASSIFICATION & GRAD-CAM
  // =========================================================================
  function setup2DControls() {
    const slider = document.getElementById("sliderCamOpacity");
    const label = document.getElementById("labelCamOpacity");
    const heatmap = document.getElementById("mriHeatmapLayer");
    const rawImg = document.getElementById("mriRawLayer");
    const btnRaw = document.getElementById("btnLayerRaw");
    const btnBlend = document.getElementById("btnLayerBlend");
    const btnHeatmap = document.getElementById("btnLayerHeatmap");

    if (slider && heatmap && label) {
      slider.addEventListener("input", (e) => {
        const val = e.target.value;
        label.textContent = `${val}%`;
        heatmap.style.opacity = val / 100;
      });
    }

    function setActiveLayerBtn(btn) {
      [btnRaw, btnBlend, btnHeatmap].forEach(b => b?.classList.remove("active"));
      btn?.classList.add("active");
    }

    btnRaw?.addEventListener("click", () => {
      setActiveLayerBtn(btnRaw);
      if (rawImg) rawImg.style.display = "block";
      if (heatmap) heatmap.style.display = "none";
    });

    btnBlend?.addEventListener("click", () => {
      setActiveLayerBtn(btnBlend);
      if (rawImg) rawImg.style.display = "block";
      if (heatmap) {
        heatmap.style.display = "block";
        heatmap.style.opacity = (slider.value / 100);
      }
    });

    btnHeatmap?.addEventListener("click", () => {
      setActiveLayerBtn(btnHeatmap);
      if (rawImg) rawImg.style.display = "none";
      if (heatmap) {
        heatmap.style.display = "block";
        heatmap.style.opacity = "1";
      }
    });
  }

  async function analyze2DSample(sampleKey) {
    const formData = new FormData();
    formData.append("sample_key", sampleKey);
    await analyze2DScan(formData);
  }

  async function analyze2DScan(formData) {
    try {
      const res = await fetch(`${API_BASE}/api/predict/2d`, {
        method: "POST",
        body: formData
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render2DResult(data);
    } catch (err) {
      alert(`2D Inference error: ${err.message}`);
    }
  }

  function render2DResult(data) {
    state.scan2d = data;

    const rawImg = document.getElementById("mriRawLayer");
    const heatmap = document.getElementById("mriHeatmapLayer");
    const emptyPrompt = document.getElementById("mriEmptyPrompt");

    if (rawImg && heatmap) {
      rawImg.src = data.original_image;
      rawImg.style.display = "block";
      if (data.gradcam_heatmap) {
        heatmap.src = data.gradcam_heatmap;
        heatmap.style.display = "block";
      }
      if (emptyPrompt) emptyPrompt.style.display = "none";
    }

    const hudLatency = document.getElementById("hudLatency");
    if (hudLatency) hudLatency.textContent = `LATENCY: ${data.inference_time_ms} ms`;

    const diagBody = document.getElementById("diagnosticBody");
    if (diagBody) {
      diagBody.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
          <span class="badge ${data.metadata.badge_class}" style="font-size: 0.95rem;">${data.metadata.label.toUpperCase()}</span>
          <span style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan); font-size: 1.2rem;">${data.confidence}%</span>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;">${data.metadata.description}</p>
        <div style="margin-top: 10px; font-size: 0.75rem; color: var(--text-dim); font-family: var(--font-mono);">
          ARCHITECTURE: ${data.model_architecture}
        </div>
      `;
    }

    const probContainer = document.getElementById("probBarsContainer");
    if (probContainer) {
      probContainer.innerHTML = "";
      for (const [cls, val] of Object.entries(data.probabilities)) {
        const item = document.createElement("div");
        item.style.marginBottom = "10px";
        item.innerHTML = `
          <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 3px;">
            <span style="font-weight: 500;">${cls.toUpperCase()}</span>
            <span style="font-family: var(--font-mono);">${val}%</span>
          </div>
          <div style="background: rgba(148, 163, 184, 0.15); height: 7px; border-radius: 4px; overflow: hidden;">
            <div style="background: var(--accent-cyan); width: ${val}%; height: 100%; border-radius: 4px;"></div>
          </div>
        `;
        probContainer.appendChild(item);
      }
    }

    const ctxMri = document.getElementById("ctxMriStatus");
    if (ctxMri) ctxMri.textContent = `${data.prediction.toUpperCase()} (${data.confidence}%)`;

    const repPred = document.getElementById("repPrediction");
    const repConf = document.getElementById("repConfidence");
    const repDesc = document.getElementById("repDescription");
    if (repPred) repPred.textContent = data.prediction.toUpperCase();
    if (repConf) repConf.textContent = `${data.confidence}%`;
    if (repDesc) repDesc.textContent = data.metadata.description;
  }

  // =========================================================================
  // 3D VOLUMETRIC SEGMENTATION & THREE.JS WEBGL
  // =========================================================================
  // Lazy-init: called the first time the 3D tab becomes visible so the
  // container already has real pixel dimensions.
  let _threeInited = false;
  function ensureThreeJs() {
    if (_threeInited || typeof THREE === "undefined") return;
    const container = document.getElementById("threejsViewport");
    if (!container) return;

    // Force the panel to have height before we measure it
    container.style.minHeight = "460px";
    const width  = container.offsetWidth  || 700;
    const height = container.offsetHeight || 460;

    // ── Scene ──────────────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    const bgColor = state.currentTheme === "light" ? 0xf1f5f9 : 0x050811;
    scene.background = new THREE.Color(bgColor);

    // Atmospheric fog to add depth without obscuring near geometry
    scene.fog = new THREE.FogExp2(bgColor, 0.0015);

    // ── Camera ─────────────────────────────────────────────────────────────
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    camera.position.set(0, 30, 200);

    // ── Renderer ───────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance"
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    container.innerHTML = "";            // clear any stale canvas
    container.appendChild(renderer.domElement);

    // ── Lights ─────────────────────────────────────────────────────────────
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0x7dd3fc, 1.4);  // soft blue key
    keyLight.position.set(120, 160, 100);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xc084fc, 0.8); // purple fill
    fillLight.position.set(-100, -80, -120);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xfbbf24, 0.6);  // amber rim
    rimLight.position.set(0, -150, 80);
    scene.add(rimLight);

    // ── OrbitControls ──────────────────────────────────────────────────────
    let controls = null;
    const OC = (typeof THREE.OrbitControls !== "undefined")
      ? THREE.OrbitControls
      : null;
    if (OC) {
      controls = new OC(camera, renderer.domElement);
      controls.enableDamping    = true;
      controls.dampingFactor    = 0.06;
      controls.autoRotate       = state.autoRotate;
      controls.autoRotateSpeed  = 1.5;
      controls.minDistance      = 60;
      controls.maxDistance      = 450;
    }

    // ── Store on state ─────────────────────────────────────────────────────
    state.threeScene    = scene;
    state.threeCamera   = camera;
    state.threeRenderer = renderer;
    state.threeControls = controls;

    window.addEventListener("resize", onWindowResize);

    (function animate() {
      requestAnimationFrame(animate);
      if (state.threeControls) state.threeControls.update();
      renderer.render(scene, camera);
    })();

    _threeInited = true;
  }

  // Keep old name for callers that might still reference it
  function initThreeJs() { ensureThreeJs(); }

  function onWindowResize() {
    const container = document.getElementById("threejsViewport");
    if (!container || !state.threeRenderer || !state.threeCamera) return;
    const width  = container.offsetWidth;
    const height = container.offsetHeight;
    if (!width || !height) return;
    state.threeCamera.aspect = width / height;
    state.threeCamera.updateProjectionMatrix();
    state.threeRenderer.setSize(width, height);
  }

  function render3DMesh(meshData) {
    ensureThreeJs();                     // safe to call even if already inited
    if (!state.threeScene || !meshData) return;

    // Clear previous meshes
    Object.values(state.threeMeshes).forEach(m => {
      state.threeScene.remove(m);
      if (m.geometry) m.geometry.dispose();
      if (m.material) m.material.dispose();
    });
    state.threeMeshes = {};

    // ── Vivid, clearly distinct region colours ──────────────────────────────
    //  WT  – Whole Tumour  : bright cyan  (outermost shell)
    //  TC  – Tumour Core   : vivid crimson (inner mass)
    //  ET  – Enhancing T.  : hot amber/gold (active hotspot)
    const regionDef = {
      WT: { color: 0x00e5ff, emissive: 0x004455, opacity: 0.42, roughness: 0.6, metalness: 0.1 },
      TC: { color: 0xff2244, emissive: 0x550010, opacity: 0.82, roughness: 0.35, metalness: 0.15 },
      ET: { color: 0xffb300, emissive: 0x553300, opacity: 0.97, roughness: 0.25, metalness: 0.3  }
    };

    let anyAdded = false;
    ["WT", "TC", "ET"].forEach(reg => {
      const regMesh = meshData[reg];
      if (!regMesh || !regMesh.vertices || regMesh.vertices.length < 9) return;

      const geom = new THREE.BufferGeometry();
      geom.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(regMesh.vertices), 3)
      );
      if (regMesh.faces && regMesh.faces.length >= 3) {
        geom.setIndex(
          new THREE.BufferAttribute(new Uint32Array(regMesh.faces), 1)
        );
      }
      geom.computeVertexNormals();

      const def = regionDef[reg];

      // Solid surface mesh
      const solidMat = new THREE.MeshStandardMaterial({
        color:      def.color,
        emissive:   def.emissive,
        transparent: true,
        opacity:    def.opacity,
        roughness:  def.roughness,
        metalness:  def.metalness,
        side:       THREE.DoubleSide,
        depthWrite: def.opacity > 0.5   // only WT is depth-transparent
      });
      const solidMesh = new THREE.Mesh(geom, solidMat);
      state.threeScene.add(solidMesh);
      state.threeMeshes[reg] = solidMesh;

      // For WT add a bright wireframe outline so it's always visible
      if (reg === "WT") {
        const wireMat = new THREE.MeshBasicMaterial({
          color:       0x22ddff,
          wireframe:   true,
          transparent: true,
          opacity:     0.18
        });
        const wireMesh = new THREE.Mesh(geom, wireMat);
        state.threeScene.add(wireMesh);
        state.threeMeshes["WT_wire"] = wireMesh;
      }

      anyAdded = true;
    });

    // Reset camera only when mesh is present
    if (anyAdded && state.threeCamera) {
      state.threeCamera.position.set(0, 30, 200);
      if (state.threeControls) state.threeControls.target.set(0, 0, 0);
    }
  }

  function setup3DControls() {
    document.querySelectorAll("[data-plane]").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-plane]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.currentPlane = btn.getAttribute("data-plane");
        updateSliceStream();
      });
    });

    const sliceSlider = document.getElementById("sliderMprSlice");
    if (sliceSlider) {
      sliceSlider.addEventListener("input", (e) => {
        state.currentSliceIdx = parseInt(e.target.value, 10);
        updateSliceStream();
      });
    }

    const cineBtn = document.getElementById("btnCinePlay");
    if (cineBtn) {
      cineBtn.addEventListener("click", () => {
        if (state.isCinePlaying) {
          clearInterval(state.cineInterval);
          state.isCinePlaying = false;
          cineBtn.textContent = "▶ Cine Loop (18 fps)";
        } else {
          state.isCinePlaying = true;
          cineBtn.textContent = "⏸ Pause Cine";
          state.cineInterval = setInterval(() => {
            const maxS = parseInt(sliceSlider.max, 10) || 150;
            state.currentSliceIdx = (state.currentSliceIdx + 1) % maxS;
            sliceSlider.value = state.currentSliceIdx;
            updateSliceStream();
          }, 55);
        }
      });
    }

    document.getElementById("btnToggleAutoRotate")?.addEventListener("click", () => {
      state.autoRotate = !state.autoRotate;
      if (state.threeControls) state.threeControls.autoRotate = state.autoRotate;
    });

    document.getElementById("btnResetCamera")?.addEventListener("click", () => {
      if (state.threeCamera && state.threeControls) {
        state.threeCamera.position.set(0, 0, 150);
        state.threeControls.target.set(0, 0, 0);
      }
    });

    ["WT", "TC", "ET"].forEach(reg => {
      document.getElementById(`btnMesh${reg}`)?.addEventListener("click", (e) => {
        const mesh = state.threeMeshes[reg];
        if (mesh) {
          mesh.visible = !mesh.visible;
          e.target.style.opacity = mesh.visible ? "1" : "0.4";
        }
      });
    });
  }

  async function loadSegmentationCase(caseKey) {
    state.current3DCaseKey = caseKey;
    const formData = new FormData();
    formData.append("case_key", caseKey);
    formData.append("plane", state.currentPlane);

    try {
      const res = await fetch(`${API_BASE}/api/segmentation/process`, {
        method: "POST",
        body: formData
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render3DCaseResults(data);
    } catch (err) {
      console.error("3D segmentation error:", err);
    }
  }

  function render3DCaseResults(data) {
    state.scan3d = data;

    const mriLayer = document.getElementById("mprMriLayer");
    const segLayer = document.getElementById("mprSegLayer");
    if (mriLayer && data.slices.mri) mriLayer.src = data.slices.mri;
    if (segLayer && data.slices.seg) segLayer.src = data.slices.seg;

    const sliceSlider = document.getElementById("sliderMprSlice");
    const sliceLabel = document.getElementById("labelMprSlice");
    if (sliceSlider) {
      sliceSlider.max = data.total_slices - 1;
      sliceSlider.value = data.slice_index;
    }
    if (sliceLabel) {
      sliceLabel.textContent = `Slice: ${data.slice_index} / ${data.total_slices}`;
    }

    if (data.meshes && data.meshes.has_mesh) {
      render3DMesh(data.meshes);
    }

    const metricsBody = document.getElementById("volumetricMetricsBody");
    if (metricsBody) {
      metricsBody.innerHTML = `
        <div style="margin-bottom: 12px;">
          <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase;">Study ID</div>
          <div style="font-family: var(--font-mono); font-weight: 700;">${data.patient_study_id}</div>
        </div>
        <div style="margin-bottom: 12px;">
          <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase;">Whole Tumor (WT) Volume</div>
          <div style="font-family: var(--font-mono); font-size: 1.4rem; font-weight: 700; color: var(--accent-cyan);">${data.volume_cm3.WT || 0} cm³</div>
        </div>
        <div style="margin-bottom: 12px;">
          <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase;">Tumor Core (TC) Volume</div>
          <div style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 700; color: var(--accent-red);">${data.volume_cm3.TC || 0} cm³</div>
        </div>
        <div style="margin-bottom: 12px;">
          <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase;">Enhancing Margin (ET) Volume</div>
          <div style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 700; color: var(--accent-gold);">${data.volume_cm3.ET || 0} cm³</div>
        </div>
      `;
    }

    const repWT = document.getElementById("repVolWT");
    const repTC = document.getElementById("repVolTC");
    const repET = document.getElementById("repVolET");
    const repVoxWT = document.getElementById("repVoxWT");
    const repVoxTC = document.getElementById("repVoxTC");
    const repVoxET = document.getElementById("repVoxET");

    if (repWT) repWT.textContent = `${data.volume_cm3.WT || 0} cm³`;
    if (repTC) repTC.textContent = `${data.volume_cm3.TC || 0} cm³`;
    if (repET) repET.textContent = `${data.volume_cm3.ET || 0} cm³`;
    if (repVoxWT) repVoxWT.textContent = (data.voxel_counts.WT || 0).toLocaleString();
    if (repVoxTC) repVoxTC.textContent = (data.voxel_counts.TC || 0).toLocaleString();
    if (repVoxET) repVoxET.textContent = (data.voxel_counts.ET || 0).toLocaleString();
  }

  async function updateSliceStream() {
    const url = `${API_BASE}/api/segmentation/slice?case_key=${state.current3DCaseKey}&plane=${state.currentPlane}&slice_idx=${state.currentSliceIdx}`;
    try {
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        const mriLayer = document.getElementById("mprMriLayer");
        const segLayer = document.getElementById("mprSegLayer");
        const sliceLabel = document.getElementById("labelMprSlice");
        if (mriLayer && data.mri_image) mriLayer.src = data.mri_image;
        if (segLayer && data.seg_image) segLayer.src = data.seg_image;
        if (sliceLabel) sliceLabel.textContent = `Slice: ${data.slice_index} / ${data.total_slices}`;
      }
    } catch (e) {
      // Ignore rapid stream aborts
    }
  }

  // =========================================================================
  // MULTIMODAL MEDICAL ASSISTANT & PDF PARSER
  // =========================================================================
  function setupChatControls() {
    const pdfInput = document.getElementById("inputPdfUpload");
    if (pdfInput) {
      pdfInput.addEventListener("change", async (e) => {
        if (e.target.files && e.target.files[0]) {
          const file = e.target.files[0];
          const formData = new FormData();
          formData.append("file", file);

          try {
            const res = await fetch(`${API_BASE}/api/pdf/extract`, {
              method: "POST",
              body: formData
            });
            if (!res.ok) throw new Error("Failed to parse PDF");
            const data = await res.json();
            state.pdfText = data.text;

            const ctxPdf = document.getElementById("ctxPdfStatus");
            if (ctxPdf) ctxPdf.textContent = `${file.name} (${data.word_count} words)`;

            const excerpt = document.getElementById("pdfSummaryExcerpt");
            if (excerpt) excerpt.textContent = data.text.slice(0, 400) + "...";

            sendChatMessage("Please provide a compassionate plain-language summary of this medical report and suggest 3 key questions for my doctor.");
          } catch (err) {
            alert(`PDF Extraction error: ${err.message}`);
          }
        }
      });
    }

    const chatForm = document.getElementById("chatForm");
    if (chatForm) {
      chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const input = document.getElementById("chatInputText");
        const msg = input.value.trim();
        if (msg) {
          input.value = "";
          sendChatMessage(msg);
        }
      });
    }
  }

  async function sendChatMessage(userText) {
    const messagesContainer = document.getElementById("chatMessages");
    if (!messagesContainer) return;

    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user";
    userBubble.textContent = userText;
    messagesContainer.appendChild(userBubble);

    const assistantBubble = document.createElement("div");
    assistantBubble.className = "chat-bubble assistant";
    assistantBubble.textContent = "Analyzing medical context...";
    messagesContainer.appendChild(assistantBubble);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    const groqKey = document.getElementById("inputGroqKey")?.value || "";
    const model = document.getElementById("selectLlmModel")?.value || "llama-3.3-70b-versatile";

    const mriContext = state.scan2d ? {
      prediction: state.scan2d.prediction,
      confidence: state.scan2d.confidence,
      description: state.scan2d.metadata?.description || "",
      volume_cm3: state.scan3d?.volume_cm3?.WT || null
    } : null;

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userText,
          conversation_history: state.chatHistory,
          mri_context: mriContext,
          pdf_context: state.pdfText,
          groq_api_key: groqKey,
          model: model
        })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";
      assistantBubble.textContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        accumulated += chunk;
        assistantBubble.innerHTML = accumulated.replace(/\n/g, "<br>");
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }

      state.chatHistory.push({ role: "user", content: userText });
      state.chatHistory.push({ role: "assistant", content: accumulated });

    } catch (err) {
      assistantBubble.textContent = `Error: ${err.message}`;
    }
  }

})();
