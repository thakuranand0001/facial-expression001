from pathlib import Path
import json
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "facial_expression.keras"
CLASSES_PATH = BASE_DIR / "models" / "classes.json"

st.set_page_config(page_title="Facial Expression Recognition", page_icon="🙂", layout="centered")
st.title("🙂 Facial Expression Recognition")
st.write("Upload a face image to view the model's predicted expression.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    resized = image.resize((64, 64))
    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    model = load_model()
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    st.subheader(f"Prediction: {class_names[index].capitalize()}")
    st.metric("Model confidence", f"{probabilities[index] * 100:.1f}%")
    st.write("Class probabilities")
    for label, score in sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True):
        st.write(f"{label.capitalize()}: {score*100:.1f}%")
        st.progress(float(score))
st.caption("Educational demo only. Predictions are not a reliable measure of a person's actual emotional state.")
