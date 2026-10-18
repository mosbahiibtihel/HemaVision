import streamlit as st
import torch
import numpy as np
import cv2
import sys

from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import build_model


# ============================================================
# Configuration
# ============================================================

CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

MODEL_PATH = "models/hemavision_efficientnet_b0.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():
    model = build_model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# Grad-CAM
# ============================================================

def generate_gradcam(model, input_tensor, predicted_class):

    activations = {}
    gradients = {}

    target_layer = model.features[-1][0]

    def forward_hook(module, inputs, output):
        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.__setitem__("value", grad)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    model.zero_grad()

    outputs = model(input_tensor)

    target_score = outputs[0, predicted_class]
    target_score.backward()

    handle.remove()

    activation = activations["value"]
    gradient = gradients["value"]

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (weights * activation).sum(dim=1)

    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()

    # Normalize
    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to image size
    cam = cv2.resize(
        cam,
        (224, 224)
    )

    # Convert to heatmap
    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return heatmap


# ============================================================
# Streamlit interface
# ============================================================

st.set_page_config(
    page_title="HemaVision",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 HemaVision")

st.subheader("Blood Cell Classification with Transfer Learning")

st.write(
    "Upload a blood-cell image to classify it using "
    "a pretrained EfficientNet-B0 model."
)

st.info(
    "Research/educational project — not a clinical diagnostic tool."
)


# ============================================================
# Load model
# ============================================================

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


st.success("Model loaded successfully.")


# ============================================================
# Upload image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a blood-cell image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Input Image")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded image",
            width="stretch"
        )

    # Prepare image
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    # Important for Grad-CAM
    input_tensor.requires_grad_(True)

    # ========================================================
    # Prediction
    # ========================================================

    model.zero_grad()

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_class = outputs.argmax(
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    predicted_name = CLASS_NAMES[predicted_class]


    # ========================================================
    # Results
    # ========================================================

    with col2:

        st.markdown("### Prediction")

        st.success(
            f"**{predicted_name}**"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ========================================================
    # Class probabilities
    # ========================================================

    st.markdown("### Class Probabilities")

    probs = probabilities[0].detach().cpu().numpy()

    probability_data = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(probability_data)


    # ========================================================
    # Grad-CAM
    # ========================================================

    st.markdown("### Grad-CAM Explanation")

    with st.spinner("Generating Grad-CAM..."):

        heatmap = generate_gradcam(
            model,
            input_tensor,
            predicted_class
        )

    # Resize original image
    original = np.array(
        image.resize((224, 224))
    )

    # Overlay
    overlay = cv2.addWeighted(
        original,
        0.5,
        heatmap,
        0.5,
        0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="Original",
            width="stretch"
        )

    with col2:
        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            width="stretch"
        )

    with col3:
        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            width="stretch"
        )

    st.caption(
        "Red/yellow regions indicate areas that contributed more "
        "strongly to the model's prediction."
    )