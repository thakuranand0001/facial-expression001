import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Facial Expression Classifier", page_icon="😊", layout="wide")
MODEL_PATH = Path("facial_expression_model.keras")
CLASSES_PATH = Path("classes.json")
IMAGE_SIZE = (64, 64)
DEFAULT_CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
EMOJI = {"angry": "😠", "disgusted": "🤢", "fearful": "😨", "happy": "😊",
         "neutral": "😐", "sad": "😢", "surprised": "😮"}

st.markdown("<style>.main-title{font-size:2.2rem;font-weight:750}.muted{color:#64748b}</style>", unsafe_allow_html=True)
with st.sidebar:
    st.title("😊 Emotion AI")
    page = st.radio("Navigation", ["Overview", "Predict Expression", "About Model"])
    st.divider()
    st.caption("CNN · 64 × 64 RGB input · 7 classes")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

if page == "Overview":
    st.markdown('<div class="main-title">Facial Expression Classification</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">A convolutional neural network demonstration using the supplied FER image dataset.</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Model", "CNN")
    cols[1].metric("Input size", "64 × 64")
    cols[2].metric("Expression classes", "7")
    st.subheader("Project workflow")
    for heading, detail in [
        ("1. Prepare dataset", "Extract the dataset ZIP. The training script combines alternate folder spellings into the same emotion label."),
        ("2. Train model", "Run train_model.py to train, validate, and evaluate the CNN."),
        ("3. Predict", "Upload a face image to view the predicted expression and class scores."),
    ]:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(detail)
    st.info("For learning and demonstration only. Facial expressions do not reliably reveal a person's internal emotional state.")

elif page == "Predict Expression":
    st.title("Upload an image")
    if not MODEL_PATH.exists():
        st.warning("Model file not found. Train the model first using the instructions in README.md.")
    else:
        try:
            model = load_model(str(MODEL_PATH))
            classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8")) if CLASSES_PATH.exists() else DEFAULT_CLASSES
            uploaded = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                left, right = st.columns([1, 1.2])
                with left:
                    st.image(image, caption="Uploaded image", use_container_width=True)
                with right:
                    if st.button("Predict expression", type="primary"):
                        resized = image.resize(IMAGE_SIZE)
                        arr = np.asarray(resized, dtype=np.float32)[None, ...]
                        scores = model.predict(arr, verbose=0)[0]
                        idx = int(np.argmax(scores))
                        st.subheader("Prediction")
                        st.markdown(f"## {EMOJI.get(classes[idx], '🙂')} {classes[idx].title()}")
                        st.write("Model score distribution")
                        for name, score in sorted(zip(classes, scores), key=lambda x: x[1], reverse=True):
                            st.write(f"{EMOJI.get(name, '•')} **{name.title()}** — {score:.1%}")
                            st.progress(float(score))
                        st.caption("Scores are model outputs, not calibrated probabilities or evidence of a person's actual feelings.")
        except Exception as exc:
            st.error(f"Could not load or run the model: {exc}")

else:
    st.title("About the model")
    st.write("The project follows the supplied introductory image-classification notebook: TensorFlow/Keras, directory-based image data, a CNN, softmax output, model saving, and class-name persistence.")
    st.markdown("**Classes:** " + ", ".join(c.title() for c in DEFAULT_CLASSES))
    st.markdown("**Input:** 64 × 64 RGB images")
    st.markdown("**Output:** Seven class scores; the highest score is displayed as the predicted label.")
    st.warning("This is an educational image classifier, not a psychological assessment tool. It may be inaccurate across lighting, pose, demographics, image quality, and datasets.")
