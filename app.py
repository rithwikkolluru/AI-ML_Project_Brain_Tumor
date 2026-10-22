import streamlit as st
import os
import requests
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from pypdf import PdfReader
import google.generativeai as genai

# ============================================================
# 1. APP CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Brain Tumor AI Diagnostic & Explainer Suite", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-glioma { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-meningioma { background-color: #f59e0b; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-pituitary { background-color: #8b5cf6; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-notumor { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .ai-disclaimer-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 15px;
        font-size: 0.9rem;
        color: #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Brain Tumor AI Diagnostic & Interactive Explainer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal AI Suite: Fine-Tuned EfficientNetB0 (93.31% Test Accuracy) + Grad-CAM Saliency Maps + Conversational AI Medical Assistant.</div>', unsafe_allow_html=True)

# Mandatory Top Banner Disclaimer
st.markdown("""
<div class="ai-disclaimer-box">
    ⚠️ <strong>AI-Generated Clinical Assistant Disclaimer:</strong> All diagnoses, classifications, heatmap visualizations, and conversational responses provided by this platform are generated autonomously by Artificial Intelligence algorithms for research and educational purposes. This system is <strong>NOT</strong> an FDA/CE-certified diagnostic device and must never substitute certified medical doctor consultations, pathology tests, or professional second opinions.
</div>
""", unsafe_allow_html=True)

# Setup model configuration
MODEL_PATH = "brain_tumor_model.keras"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DESCRIPTIONS = {
    "glioma": "Glioma: Tumor originating in glial cells of brain tissue. Requires specialized neuro-oncology evaluation.",
    "meningioma": "Meningioma: Tumor arising from the meninges (membranes enveloping the brain and spinal cord).",
    "notumor": "No Tumor Detected: Scan demonstrates normal cerebral architecture with no evident neoplastic lesions.",
    "pituitary": "Pituitary Tumor: Mass lesion in the pituitary sella region affecting endocrine/visual pathways."
}

@st.cache_resource
def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.sidebar.error(f"❌ Model load error: {e}")
            return None
    else:
        st.sidebar.warning(f"⚠️ {MODEL_PATH} not found!")
        return None

my_cnn_model = load_cnn_model()

# ============================================================
# SIDEBAR CONTROLS & API KEYS
# ============================================================
st.sidebar.header("🔑 AI Engine Configuration")
st.sidebar.info("Model: **EfficientNetB0 (M5 GPU Trained)**\nAccuracy: **93.31%** (1,600 test scans)\nClasses: Glioma, Meningioma, No Tumor, Pituitary")

llm_mode = st.sidebar.selectbox("LLM Provider", ["Online (Gemini Cloud)", "Offline (Local Gemma 4 / Ollama)"])

gemini_key = st.sidebar.text_input(
    "Gemini API Key", 
    type="password", 
    value=os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")), 
    placeholder="Enter AIza... key"
)

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]

selected_gemini_model = st.sidebar.selectbox("Active Model", GEMINI_MODELS, index=0)

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state["chat_messages"] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🛡️ Safety & Privacy
- Your API keys are strictly retained in your local session.
- Medical scans are evaluated locally in memory on your device.
""")

# Initialize session state for persistent interactive chat
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "👋 Hello! I am your AI Medical Explainer Assistant. Upload an MRI scan or a medical PDF report, or ask me any question about brain tumor types, symptoms, terminology, or diagnostic procedures.\n\n*Note: I am an AI assistant and not a medical doctor. Always consult your physician regarding clinical decisions.*"}
    ]
if "last_mri_result" not in st.session_state:
    st.session_state["last_mri_result"] = None
if "last_pdf_text" not in st.session_state:
    st.session_state["last_pdf_text"] = None

# ============================================================
# 2. IMAGE PREPROCESSING & GRAD-CAM ENGINE
# ============================================================
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((224, 224))
    # Raw pixel values in [0, 255] as expected by EfficientNetB0
    img_array = np.array(img_resized, dtype=np.float32)
    return img_array, img

def generate_gradcam(img_array, model):
    try:
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # 1. Forward pass prediction
        preds = model.predict(img_tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        
        # 2. Extract Grad-CAM using top convolutional layer of the base model
        try:
            base_model = model.get_layer("efficientnetb0")
            target_layer = base_model.get_layer("top_conv")
            
            grad_base = tf.keras.models.Model(
                inputs=base_model.inputs,
                outputs=[target_layer.output, base_model.output]
            )
            
            head_layers = model.layers[2:]
            
            with tf.GradientTape() as tape:
                conv_outputs, base_out = grad_base(img_tensor)
                tape.watch(conv_outputs)
                x = base_out
                for layer in head_layers:
                    x = layer(x)
                loss = x[:, class_idx]
                
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            heatmap_np = heatmap.numpy()
        except Exception:
            heatmap_np = None
            
        return heatmap_np, class_idx, confidence, preds
    except Exception as e:
        st.error(f"Inference error: {str(e)}")
        return None, None, None, None

# ============================================================
# 3. PDF REPORT EXTRACTION
# ============================================================
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

# ============================================================
# 4. CHATBOT STREAMING COMPLETION ENGINE WITH FALLBACKS
# ============================================================
def stream_chatbot_response(user_prompt, conversation_history, mri_context=None, pdf_context=None, use_offline=False, target_model="gemini-2.5-flash"):
    system_instruction = """You are a compassionate, knowledgeable AI Medical Education Assistant specializing in neuro-oncology and brain MRI scans.
Your goal is to help patients and caregivers understand their medical scans, terms, and diagnostic reports in plain, supportive language.

CRITICAL MEDICAL & LEGAL RULES:
1. You must ALWAYS include a clear reminder that your response is generated by AI for educational understanding only, and does not replace consultation with a qualified neurologist, oncologist, or primary physician.
2. Never offer definitive clinical prognoses, prescriptions, or treatment regimens. Frame recommendations as questions or talking points for their doctor.
3. Be reassuring, empathetic, objective, and clear without causing undue panic."""

    # Build context prompt
    context_sections = []
    if mri_context:
        context_sections.append(f"PATIENT'S RECENT MRI SCAN RESULT (AI CNN Analysis):\n- Tumor Classification: {mri_context.get('class')}\n- Model Confidence: {mri_context.get('confidence')}\n- Description: {mri_context.get('desc')}")
    if pdf_context:
        context_sections.append(f"PATIENT'S UPLOADED CLINICAL/RADIOLOGY REPORT TEXT:\n---\n{pdf_context[:2500]}\n---")

    context_str = "\n\n".join(context_sections)
    if context_str:
        augmented_prompt = f"ACTIVE PATIENT DATA CONTEXT:\n{context_str}\n\nUSER QUESTION: {user_prompt}"
    else:
        augmented_prompt = user_prompt

    if use_offline:
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "gemma4:12b",
            "prompt": f"{system_instruction}\n\n{augmented_prompt}",
            "stream": True,
            "options": {"temperature": 0.25}
        }
        try:
            headers = {"Content-Type": "application/json"}
            res = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
            if res.status_code == 200:
                for line in res.iter_lines():
                    if line:
                        decoded_line = json.loads(line.decode("utf-8"))
                        yield decoded_line.get("response", "")
            else:
                yield f"⚠️ Ollama Server returned status {res.status_code}. Ensure Ollama is running (`ollama run gemma4:12b`)."
        except Exception as e:
            yield f"⚠️ Ollama Connection Error: {str(e)}. Switch to 'Online (Gemini Cloud)' in the left sidebar if you don't have local Ollama running."
    else:
        if not gemini_key or len(gemini_key.strip()) < 10:
            yield "⚠️ **Gemini API Key Required**: Please enter your Google Gemini API key in the left sidebar to enable the conversational AI doctor assistant. (Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey))."
            return

        try:
            genai.configure(api_key=gemini_key.strip())
            model_client = genai.GenerativeModel(
                model_name=target_model,
                system_instruction=system_instruction,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.25,
                    max_output_tokens=800,
                )
            )

            history = []
            for msg in conversation_history[-6:]:
                role = msg["role"]
                gemini_role = "model" if role == "assistant" else "user"
                history.append({"role": gemini_role, "parts": [msg["content"]]})

            chat = model_client.start_chat(history=history)
            response = chat.send_message(augmented_prompt, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"⚠️ Gemini API Error: {str(e)}. Please verify your API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)."

# ============================================================
# 5. USER INTERFACE (Tabs Layout)
# ============================================================
tab_scan, tab_report, tab_chat = st.tabs([
    "🖼️ 1. MRI Scan & Heatmap", 
    "📄 2. Medical PDF Report", 
    "💬 3. AI Medical Chatbot Assistant"
])

# ------------------------------------------------------------
# TAB 1: MRI SCAN ANALYSIS
# ------------------------------------------------------------
with tab_scan:
    st.markdown("#### Upload Brain MRI Scan for Deep Learning Classification & Localization")
    col_upload, col_display = st.columns([1, 1.2])
    
    with col_upload:
        uploaded_image = st.file_uploader("Select MRI Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="mri_file")
        if uploaded_image is not None:
            if my_cnn_model is None:
                st.error("⚠️ CNN Model is not loaded. Ensure 'brain_tumor_model.keras' exists.")
            else:
                img_array, original_img = preprocess_image(uploaded_image)
                
                with st.spinner("Analyzing MRI pixels with fine-tuned EfficientNetB0..."):
                    heatmap, class_idx, confidence, all_preds = generate_gradcam(img_array, my_cnn_model)
                    
                if class_idx is not None:
                    predicted_class = CLASSES[class_idx]
                    st.session_state["last_mri_result"] = {
                        "class": predicted_class.upper(),
                        "confidence": f"{confidence:.2%}",
                        "desc": CLASS_DESCRIPTIONS[predicted_class]
                    }
                    
                    # Highlighted Diagnosis Card
                    badge_class = f"badge-{predicted_class}"
                    st.markdown(f"""
                    <div class="metric-card">
                        <span style="font-size: 1.05rem; color: #94a3b8;">Predicted Classification:</span> 
                        <span class="{badge_class}" style="font-size: 1.15rem; margin-left: 8px;">{predicted_class.upper()}</span>
                        <h2 style="margin-top: 8px; margin-bottom: 4px; color: #38bdf8;">Confidence: {confidence:.2%}</h2>
                        <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 6px;">{CLASS_DESCRIPTIONS[predicted_class]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 📊 Probability Breakdown Across Classes")
                    for c_name, prob in zip(CLASSES, all_preds):
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            st.write(f"**{c_name.capitalize()}**")
                        with c2:
                            st.progress(float(prob), text=f"{prob:.2%}")
                            
                    st.info("💡 Tip: Switch to the **💬 3. AI Medical Chatbot Assistant** tab to ask questions specifically about this scan result!")

    with col_display:
        if uploaded_image is not None and "class_idx" in locals() and class_idx is not None:
            st.markdown("##### Visual Inspection & Tumor Localization (Grad-CAM)")
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(original_img, caption="Original Input Scan", use_container_width=True)
            with img_c2:
                if heatmap is not None:
                    orig_np = np.array(original_img.resize((224, 224)))
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    
                    superimposed = heatmap_color * 0.45 + orig_np * 0.55
                    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                    st.image(superimposed, caption="Grad-CAM Tumor Localization Map", use_container_width=True)
                else:
                    st.warning("Localization heatmap could not be resolved.")
        else:
            st.markdown("👈 *Please upload an MRI image on the left to view diagnosis and Grad-CAM saliency heatmaps.*")

# ------------------------------------------------------------
# TAB 2: MEDICAL PDF REPORT PARSER & EXPLAINER
# ------------------------------------------------------------
with tab_report:
    st.markdown("#### Upload Radiology or Clinical PDF Report for Plain-Language Summary")
    uploaded_pdf = st.file_uploader("Upload Clinical Report (.pdf)", type=["pdf"], key="pdf_file")
    
    if uploaded_pdf is not None:
        with st.spinner("Extracting text from PDF report..."):
            extracted_text = extract_text_from_pdf(uploaded_pdf)
            
        if extracted_text:
            st.session_state["last_pdf_text"] = extracted_text
            st.success("✅ Clinical report successfully extracted and loaded into AI Assistant context!")
            
            with st.expander("📄 View Extracted Report Text", expanded=False):
                st.text(extracted_text[:1500] + ("..." if len(extracted_text) > 1500 else ""))
                
            st.markdown("##### 🩺 Automated Clinical Summary & Medical Term Breakdown")
            is_offline = (llm_mode == "Offline (Local Gemma 4 / Ollama)")
            
            auto_summary_prompt = "Please provide a compassionate summary of this uploaded medical report, break down difficult radiological terms into everyday analogies, and recommend 3 questions for the doctor."
            
            with st.chat_message("assistant"):
                st.write_stream(stream_chatbot_response(
                    auto_summary_prompt,
                    conversation_history=[],
                    mri_context=st.session_state["last_mri_result"],
                    pdf_context=extracted_text,
                    use_offline=is_offline,
                    target_model=selected_gemini_model
                ))
                
            st.info("💡 You can now chat interactively in the **💬 3. AI Medical Chatbot Assistant** tab to ask detailed follow-up questions about this report!")

# ------------------------------------------------------------
# TAB 3: CONVERSATIONAL AI MEDICAL CHATBOT
# ------------------------------------------------------------
with tab_chat:
    st.markdown("#### 💬 Interactive AI Medical Assistant")
    st.caption("Ask questions about your uploaded MRI scan, medical PDF report, tumor terminology, symptoms, or diagnostic tests.")
    
    # Active Context Indicator Cards
    ctx_col1, ctx_col2 = st.columns(2)
    with ctx_col1:
        if st.session_state["last_mri_result"]:
            res = st.session_state["last_mri_result"]
            st.success(f"🖼️ **Attached MRI Context**: {res['class']} (Confidence: {res['confidence']})")
        else:
            st.info("🖼️ No MRI scan attached yet. (Upload in Tab 1)")
            
    with ctx_col2:
        if st.session_state["last_pdf_text"]:
            st.success(f"📄 **Attached Medical PDF Report**: {len(st.session_state['last_pdf_text'].split())} words loaded")
        else:
            st.info("📄 No medical PDF report attached yet. (Upload in Tab 2)")
            
    # Display Chat Conversation History
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat Input Box
    user_query = st.chat_input("Ask a question about your scan, report, or medical terms...")
    
    if user_query:
        # Display user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Stream Assistant Response
        is_offline = (llm_mode == "Offline (Local Gemma 4 / Ollama)")
        with st.chat_message("assistant"):
            response_generator = stream_chatbot_response(
                user_prompt=user_query,
                conversation_history=st.session_state["chat_messages"][:-1],
                mri_context=st.session_state["last_mri_result"],
                pdf_context=st.session_state["last_pdf_text"],
                use_offline=is_offline,
                target_model=selected_gemini_model
            )
            full_response = st.write_stream(response_generator)
            
        st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})