# app.py

import json
from pathlib import Path

import gdown
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import time
import os
import json
from pathlib import Path



# ====================== CONFIG ======================

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tin_shed_resnet18.pth"
CLASS_IDX_PATH = Path("class_indices.json")

# 🔴 REPLACE WITH YOUR REAL GOOGLE DRIVE FILE ID
import streamlit as st  # already imported at top

# Read from Streamlit secrets when deployed; fallback for local testing
# Google Drive model file ID

DRIVE_URL = "https://drive.google.com/file/d/1Z7kWqg6hcXfT7oRe0X9r83NXy_1rB4Z6/view?usp=sharing"
  # <-- your real ID




# For Streamlit Cloud it's safer to use CPU
DEVICE = "cpu"

THRESHOLD = 0.80  # 80% confidence required


# ================== MODEL DOWNLOAD ==================

def download_model():
    """Force download model file from Google Drive."""
    MODEL_DIR.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()  # remove any corrupted version

    st.info("📥 Downloading model from Google Drive (this may take a moment)...")
    gdown.download(DRIVE_URL, str(MODEL_PATH), quiet=False, fuzzy=True)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.write(f"✅ Model downloaded. Size: {size_mb:.2f} MB")


def ensure_model_file():
    """Ensure model file exists and is loadable, otherwise re-download."""
    if not MODEL_PATH.exists():
        download_model()
        return

    # Try loading once; if it fails, re-download
    try:
        _ = torch.load(MODEL_PATH, map_location="cpu")
    except Exception as e:
        st.warning(f"Model file seems corrupted, re-downloading. Error: {e}")
        download_model()



# ================== UTIL FUNCTIONS ==================

@st.cache_resource
def load_class_indices():
    """Load mapping {0: 'damaged', 1: 'semi_damaged', ...}."""
    with CLASS_IDX_PATH.open("r") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


@st.cache_resource
@st.cache_resource
def load_model():
    """Ensure model is present, then load ResNet18 with trained weights."""
    ensure_model_file()

    class_indices = load_class_indices()
    num_classes = len(class_indices)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model, class_indices



def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def preprocess_image(image: Image.Image):
    """Preprocess PIL image → model tensor."""
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)


# ================== GRAD-CAM IMPLEMENTATION ==================

class GradCAM:
    """
    Simple Grad-CAM for ResNet18.
    We hook into layer4 (last conv block) by default.
    """

    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        target_layer = dict([*self.model.named_modules()])[target_layer_name]

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, index=None):
        """
        x: preprocessed tensor [1, C, H, W]
        index: class index; if None uses predicted class
        """
        self.model.zero_grad()
        output = self.model(x)

        if index is None:
            index = torch.argmax(output, dim=1).item()

        target = output[0, index]
        target.backward()

        # Global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [C,1,1]
        cam = (weights * self.activations).sum(dim=1)[0]         # [H,W]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def build_cam_visuals(original_pil: Image.Image, cam: np.ndarray, alpha=0.5):
    """
    Create:
      - heatmap only (color)
      - overlay (heatmap + original)
    Use smoother colormap and light blur for nicer visualization.
    """
    img = np.array(original_pil)  # HxWx3, RGB

    # Resize cam to image size
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Slight blur for smoother heatmap
    cam_blurred = cv2.GaussianBlur(cam_resized, (9, 9), 0)

    # Use modern colormap (TURBO)
    heatmap = cv2.applyColorMap((cam_blurred * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return heatmap, overlay


# ================== PREDICTION + CAM + THRESHOLD ==================

def predict_with_cam(image: Image.Image, model, class_indices):
    """
    Predict class + confidence + prob vector + Grad-CAM map.
    Returns:
        label, confidence, probs (np array), is_confident, cam (2D array 0–1)
    """
    tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    label = class_indices[pred_idx]
    confidence = float(probs[pred_idx].item())
    is_confident = confidence >= THRESHOLD

    gradcam = GradCAM(model, target_layer_name="layer4")
    cam = gradcam(tensor)  # [H,W], 0–1

    return label, confidence, probs.cpu().numpy(), is_confident, cam


# ================== RECOMMENDATION LOGIC ==================

def get_recommendation(label: str, is_confident: bool) -> str:
    """
    Return human-readable recommendation text based on prediction.
    Ensures non_damaged is NOT misclassified as damaged.
    """
    label_lower = label.lower().strip()

    if not is_confident:
        return (
            "The model is not very confident about this prediction. "
            "Try capturing another clear image of the tin shed from closer distance, "
            "avoiding heavy shadows or obstructions. If corrosion is suspected, "
            "consider manual inspection by a technician."
        )

    # ----- Damaged -----
    if label_lower == "damaged":
        return (
            "High level of corrosion detected. Maintenance should be scheduled immediately. "
            "Replacement or reinforcement of affected tin sheets is advised, and structural "
            "weakness or water leakage should be thoroughly checked."
        )

    # ----- Semi-Damaged -----
    if label_lower == "semi_damaged":
        return (
            "Moderate corrosion detected. Regular monitoring is recommended. "
            "Cleaning the affected area, applying an anti-corrosion coating, "
            "and planning preventive maintenance will help slow deterioration."
        )

    # ----- Non-Damaged -----
    if label_lower == "non_damaged":
        return (
            "No significant corrosion detected. The tin sheet appears to be in good condition. "
            "Routine inspections and basic cleaning are recommended to prevent future rust formation."
        )

    # ----- Fallback -----
    return "Prediction received, but no recommendation rule matches this class."



# ====================== STREAMLIT APP ======================

def main():
    st.set_page_config(
        page_title="Tin Shed Corrosion Detection",
        page_icon="🛠️",
        layout="centered",
    )

    st.title("🛠️ Tin Shed Corrosion Detection with Saliency Heatmap")
    st.write(
        "Upload an image of a tin shed. "
        "The model classifies it as **damaged**, **semi_damaged**, or **non_damaged** and shows a "
        "Grad-CAM heatmap highlighting the regions that most influenced the decision.\n\n"
        f"If the highest confidence is below **{THRESHOLD*100:.0f}%**, "
        "the prediction is marked as **uncertain** to avoid illogical results on unrelated images."
    )

    model, class_indices = load_model()

    # 🔹 Controls for visualization
    with st.expander("Visualization settings", expanded=True):
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            alpha = st.slider(
                "Heatmap opacity (higher = stronger color)",
                min_value=0.2,
                max_value=0.9,
                value=0.5,
                step=0.05,
            )
        with col_ctrl2:
            show_heatmap_only = st.checkbox("Show pure heatmap", value=True)
            show_overlay = st.checkbox("Show overlay heatmap", value=True)

    uploaded_file = st.file_uploader(
        "Upload a tin-shed image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Input & Corrosion Heatmap")

        # Three columns: Original | Heatmap | Overlay
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Running model & generating heatmap..."):
                label, confidence, probs, is_confident, cam = predict_with_cam(
                    image, model, class_indices
                )

            # Decide whether to show Grad-CAM when uncertain
            show_heatmap = False

            if is_confident:
                show_heatmap = True
                st.success(
                    f"Prediction: **{label}** ✅\n\n"
                    f"Confidence: **{confidence * 100:.2f}%** "
                    f"(above {THRESHOLD*100:.0f}% threshold)"
                )
            else:
                st.warning(
                    f"Prediction is **UNCERTAIN**.\n\n"
                    f"Top class: **{label}** with confidence **{confidence * 100:.2f}%**, "
                    f"which is below the {THRESHOLD*100:.0f}% threshold.\n\n"
                    "This may not be a proper tin-shed image or may be outside the training distribution."
                )
                show_heatmap = st.checkbox(
                    "Show Grad-CAM heatmap even though prediction is uncertain",
                    value=False,
                )

            # 🔹 Recommendation section (always shown)
                        # 🔹 Recommendation section (always shown, color-coded with icons)
            recommendation = get_recommendation(label, is_confident)
            label_lower = label.lower().strip()

            st.subheader("Recommended action")

            if not is_confident:
                # Uncertain prediction → yellow warning card
                st.warning(f"⚠️ {recommendation}")
            else:
                if label_lower == "damaged":
                    # High corrosion → red card
                    st.error(f"❗ {recommendation}")
                elif label_lower == "semi_damaged":
                    # Moderate corrosion → yellow card
                    st.warning(f"⚠️ {recommendation}")
                elif label_lower == "non_damaged":
                    # Healthy roof → green card
                    st.success(f"✔️ {recommendation}")
                else:
                    # Fallback → neutral info card
                    st.info(f"ℹ️ {recommendation}")


            # 🔹 Show heatmap visuals if enabled
            if show_heatmap:
                heatmap_img, overlay_img = build_cam_visuals(image, cam, alpha=alpha)

                if show_heatmap_only:
                    with col2:
                        st.image(
                            heatmap_img,
                            caption="Heatmap only (model attention)",
                            use_container_width=True,
                        )
                if show_overlay:
                    with col3:
                        st.image(
                            overlay_img,
                            caption="Overlay (heatmap on top of image)",
                            use_container_width=True,
                        )

                st.caption(
                    "Red / Yellow regions indicate high model attention (higher chance of corrosion). "
                    "Blue regions have low influence."
                )

            # Probabilities (always in an expander)
                        # ================== CLASS PROBABILITIES WITH CHARTS ==================
                        # ================== CLASS PROBABILITIES WITH CHARTS ==================
            with st.expander("Class probabilities & details", expanded=True):

                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go
                import time

                st.subheader("Class probabilities")

                # ---- Controls for visualization style ----
                col_ctrl_a, col_ctrl_b = st.columns(2)
                with col_ctrl_a:
                    anim_speed = st.slider(
                        "Animation speed (seconds per step)",
                        min_value=0.02,
                        max_value=0.30,
                        value=0.08,
                        step=0.02,
                    )
                with col_ctrl_b:
                    bar_orientation = st.radio(
                        "Bar orientation",
                        ["Vertical", "Horizontal"],
                        index=0,
                    )

                # ---- Build probability DataFrame in fixed class order ----
                idx_sorted = sorted(class_indices.keys())
                class_names = [class_indices[i] for i in idx_sorted]
                prob_values = [float(probs[i] * 100) for i in idx_sorted]  # %

                df_probs = pd.DataFrame({
                    "Class": class_names,
                    "Probability": prob_values,
                })

                # Custom color palette
                palette = px.colors.qualitative.Set2

                # ---- Side-by-side layout: bar chart | pie chart ----
                col_bar, col_pie = st.columns(2)

                # ---------- Animated BAR CHART ----------
                with col_bar:
                    st.markdown("#### Bar view")

                    placeholder_bar = st.empty()
                    for scale in [0.2, 0.4, 0.6, 0.8, 1.0]:
                        df_anim = df_probs.copy()
                        df_anim["Probability"] = df_probs["Probability"] * scale

                        if bar_orientation == "Horizontal":
                            fig_bar = px.bar(
                                df_anim,
                                x="Probability",
                                y="Class",
                                orientation="h",
                                range_x=[0, 100],
                                text="Probability",
                                color="Class",
                                color_discrete_sequence=palette,
                                labels={"Probability": "Probability (%)"},
                            )
                        else:
                            fig_bar = px.bar(
                                df_anim,
                                x="Class",
                                y="Probability",
                                range_y=[0, 100],
                                text="Probability",
                                color="Class",
                                color_discrete_sequence=palette,
                                labels={"Probability": "Probability (%)"},
                            )

                        fig_bar.update_traces(
                            texttemplate="%{text:.1f}%",
                            textposition="outside",
                            width=0.5,
                        )
                        fig_bar.update_layout(
                            yaxis_title="Probability (%)" if bar_orientation == "Vertical" else "",
                            xaxis_title="" if bar_orientation == "Vertical" else "Probability (%)",
                            margin=dict(l=10, r=10, t=40, b=10),
                        )

                        placeholder_bar.plotly_chart(fig_bar, use_container_width=True)
                        time.sleep(anim_speed)

                # ---------- PIE / DONUT CHART ----------
                with col_pie:
                    st.markdown("#### Pie view")

                    fig_pie = px.pie(
                        df_probs,
                        names="Class",
                        values="Probability",
                        hole=0.4,
                        color="Class",
                        color_discrete_sequence=palette,
                    )
                    fig_pie.update_traces(
                        textposition="inside",
                        textinfo="label+percent",
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

                # ---------- Gauge for predicted class ----------
                st.markdown("#### Predicted class gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(confidence * 100),
                    title={'text': f"{label} (confidence %)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2E86AB"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffe6e6"},
                            {'range': [50, 80], 'color': "#fff5cc"},
                            {'range': [80, 100], 'color': "#e0ffe6"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': THRESHOLD * 100,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=250,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.caption(
                    "Probabilities come from a softmax over the model outputs and sum to 100% across classes."
                )



if __name__ == "__main__":
    main()
