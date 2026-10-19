"""
Brain Tumor AI Diagnostic Suite — 2D Neural Network & Grad-CAM Engine
EfficientNetB0 Deep Learning Model (93.31% Test Accuracy, 100% Normal Brain Recall)
"""

import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_META = {
    "glioma": {
        "label": "Glioma",
        "category": "Tumor Indicated",
        "badge_class": "badge-glioma",
        "color": "#ef4444",
        "description": "Intra-axial glial origin neoplasm affecting supportive cerebral tissue. Requires specialized neuro-oncology evaluation.",
        "risk_level": "High Priority"
    },
    "meningioma": {
        "label": "Meningioma",
        "category": "Tumor Indicated",
        "badge_class": "badge-meningioma",
        "color": "#f59e0b",
        "description": "Extra-axial dural-based lesion originating from the arachnoid layer of the meninges enveloping the brain.",
        "risk_level": "Moderate to High Priority"
    },
    "notumor": {
        "label": "No Tumor Detected",
        "category": "Clear Scan (Control)",
        "badge_class": "badge-notumor",
        "color": "#10b981",
        "description": "Scan demonstrates normal cerebral architecture with no evident mass effect, midline shift, or neoplastic lesion.",
        "risk_level": "Normal Control"
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "category": "Tumor Indicated",
        "badge_class": "badge-pituitary",
        "color": "#8b5cf6",
        "description": "Mass lesion in the sellar or parasellar region, potentially compromising the optic chiasm and endocrine hypothalamic axis.",
        "risk_level": "Specialist Evaluation"
    }
}

IMG_SIZE = (224, 224)


class BrainTumorClassifier2D:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self.grad_base = None
        self.head_layers = []
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            print(f"[WARNING] Model file not found at {self.model_path}")
            return

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            print(f"[INFO] Successfully loaded 2D classification model: {self.model_path.name}")
            
            # Setup Grad-CAM sub-model
            try:
                base_model = self.model.get_layer("efficientnetb0")
                target_layer = base_model.get_layer("top_conv")
                self.grad_base = tf.keras.models.Model(
                    inputs=base_model.inputs,
                    outputs=[target_layer.output, base_model.output]
                )
                self.head_layers = self.model.layers[2:]
            except Exception as e:
                print(f"[WARNING] Could not initialize Grad-CAM graph: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to load 2D model: {e}")

    def predict(self, image_bytes: bytes, filename: str = "scan.jpg") -> Dict[str, Any]:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_width, original_height = img.size
        img_resized = img.resize(IMG_SIZE)
        img_np = np.array(img_resized, dtype=np.float32)
        img_tensor = np.expand_dims(img_np, axis=0)

        # In case model is unavailable, return structured fallback
        if self.model is None:
            raise RuntimeError("2D Classification model is not loaded.")

        # Forward pass prediction
        preds = self.model.predict(img_tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        predicted_class = CLASSES[class_idx]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        probabilities = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}

        # Generate Grad-CAM Saliency Map
        heatmap_b64, overlay_b64 = self._generate_gradcam_overlay(img_tensor, img_resized, class_idx)

        # Buffer original preview as Base64
        buf_orig = io.BytesIO()
        img_resized.save(buf_orig, format="PNG")
        original_b64 = "data:image/png;base64," + base64.b64encode(buf_orig.getvalue()).decode("utf-8")

        return {
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities,
            "inference_time_ms": elapsed_ms,
            "metadata": CLASS_META[predicted_class],
            "original_dimensions": [original_width, original_height],
            "original_image": original_b64,
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": overlay_b64,
            "model_architecture": "EfficientNetB0 (ImageNet Fine-Tuned, 2-Stage Transfer Learning)"
        }

    def _generate_gradcam_overlay(
        self, img_tensor: np.ndarray, original_pil: Image.Image, class_idx: int
    ) -> Tuple[Optional[str], Optional[str]]:
        if self.grad_base is None or not self.head_layers:
            return None, None

        try:
            with tf.GradientTape() as tape:
                conv_outputs, base_out = self.grad_base(img_tensor)
                tape.watch(conv_outputs)
                x = base_out
                for layer in self.head_layers:
                    x = layer(x)
                loss = x[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            heatmap_np = heatmap.numpy()

            # Resize heatmap to 224x224
            heatmap_resized = cv2.resize(heatmap_np, IMG_SIZE)
            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            # Apply COLORMAP_JET
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # Create Superimposed overlay (50/50 blend)
            orig_np = np.array(original_pil)
            superimposed = np.clip(heatmap_rgb * 0.45 + orig_np * 0.55, 0, 255).astype(np.uint8)

            # Heatmap with alpha channel for transparent overlay
            heatmap_rgba = np.dstack((heatmap_rgb, (heatmap_resized * 200).astype(np.uint8)))

            # Buffer Heatmap
            buf_hm = io.BytesIO()
            Image.fromarray(heatmap_rgba, mode="RGBA").save(buf_hm, format="PNG")
            heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf_hm.getvalue()).decode("utf-8")

            # Buffer Overlay
            buf_ov = io.BytesIO()
            Image.fromarray(superimposed, mode="RGB").save(buf_ov, format="PNG")
            overlay_b64 = "data:image/png;base64," + base64.b64encode(buf_ov.getvalue()).decode("utf-8")

            return heatmap_b64, overlay_b64
        except Exception as e:
            print(f"[WARNING] Grad-CAM generation failed: {e}")
            return None, None
