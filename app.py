import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load EfficientNet-V2-S model
# ------------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    state = torch.load("efficientnet_v2_s_best.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)


# ------------------------------
# Prediction function
# ------------------------------
def predict(model, image):
    tensor = preprocess(image).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()

    pred_class = "POSITIVE (Abnormal)" if prob >= 0.5 else "NEGATIVE (Normal)"
    return pred_class, prob


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="MURA X-Ray Classifier", layout="wide")

st.title("🩻 MURA X-Ray Binary Classifier")
st.write("Upload an X-ray image (Shoulder / Wrist / Elbow / Hand / Finger / Forearm / Humerus).")

model = load_model()

uploaded_file = st.file_uploader("Upload X-Ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Predict"):
        pred_class, prob = predict(model, image)

        st.subheader(f"**Prediction:** {pred_class}")



