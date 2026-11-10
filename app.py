
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Facial Expression Classifier",
    page_icon="😀",
    layout="wide"
)

# --------------------------------
# MODEL CONFIGURATION
# --------------------------------

MODEL_PATH = Path("facial_expression_model.keras")
CLASSES_PATH = Path("classes.json")

IMAGE_SIZE = (64, 64)

DEFAULT_CLASSES = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

EMOJI = {
    "angry": "😠",
    "disgusted": "🤢",
    "fearful": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprised": "😲"
}

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 750;
}
.muted {
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:
    st.title("😊 Emotion AI")

    page = st.radio(
        "Navigation",
        ["Overview", "Predict Expression", "About Model"]
    )

    st.divider()
    st.caption("CNN | 64 × 64 RGB | 7 Classes")

# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# --------------------------------
# OVERVIEW PAGE
# --------------------------------

if page == "Overview":

    st.markdown(
        '<div class="main-title">'
        'Facial Expression Classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">'
        'Deep Learning Based Facial Expression Recognition System'
        '</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "CNN")
    col2.metric("Input Size", "64 × 64")
    col3.metric("Expression Classes", "7")

    st.subheader("Project Workflow")

    steps = [
        ("1. Dataset Preparation",
         "Prepare and organize facial expression images."),

        ("2. Model Training",
         "Train the CNN model using TensorFlow and Keras."),

        ("3. Expression Prediction",
         "Upload a face image to predict its expression.")
    ]

    for heading, detail in steps:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(detail)

    st.info(
        "This application is designed for educational "
        "and demonstration purposes."
    )

# --------------------------------
# PREDICT EXPRESSION PAGE
# --------------------------------

elif page == "Predict Expression":

    st.title("📸 Upload Facial Image")

    if not MODEL_PATH.exists():

        st.warning(
            "Model file not found! "
            "Please train the model first."
        )

    else:

        try:

            model = load_model()

            if CLASSES_PATH.exists():
                classes = json.loads(
                    CLASSES_PATH.read_text(encoding="utf-8")
                )
            else:
                classes = DEFAULT_CLASSES

            uploaded_file = st.file_uploader(
                "Upload JPG, JPEG or PNG Image",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                col1, col2 = st.columns([1, 1.2])

                with col1:
                    st.image(
                        image,
                        caption="Uploaded Image",
                        use_container_width=True
                    )

                with col2:

                    if st.button(
                        "🔍 Predict Expression",
                        type="primary"
                    ):

                        resized_image = image.resize(
                            IMAGE_SIZE
                        )

                        image_array = np.asarray(
                            resized_image,
                            dtype=np.float32
                        )

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        # IMPORTANT:
                        # If training used external normalization,
                        # uncomment the next line.
                        # image_array = image_array / 255.0

                        predictions = model.predict(
                            image_array,
                            verbose=0
                        )[0]

                        if len(predictions) != len(classes):
                            st.error(
                                "Class labels do not match model output."
                            )
                        else:

                            predicted_index = int(
                                np.argmax(predictions)
                            )

                            predicted_expression = classes[
                                predicted_index
                            ]

                            st.subheader("Prediction Result")

                            st.markdown(
                                f"## {EMOJI.get(predicted_expression, '🙂')} "
                                f"{predicted_expression.title()}"
                            )

                            st.write("Expression Score Distribution")

                            sorted_results = sorted(
                                zip(classes, predictions),
                                key=lambda x: x[1],
                                reverse=True
                            )

                            for emotion, score in sorted_results:

                                st.write(
                                    f"{EMOJI.get(emotion, '•')} "
                                    f"**{emotion.title()}** — {score:.2%}"
                                )

                                st.progress(
                                    float(np.clip(score, 0, 1))
                                )

                            st.caption(
                                "Model scores are not reliable measurements "
                                "of a person's actual emotional state."
                            )

        except Exception as error:

            st.error(
                f"Prediction Error: {error}"
            )

# --------------------------------
# ABOUT MODEL PAGE
# --------------------------------

else:

    st.title("ℹ️ About the Model")

    st.write(
        "This project uses a Convolutional Neural Network (CNN) "
        "to classify facial expression images."
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - TensorFlow
    - Keras
    - CNN (Convolutional Neural Network)
    - NumPy
    - Pillow
    - Streamlit
    - GitHub
    """)

    st.subheader("Expression Classes")

    st.write(", ".join(
        emotion.title() for emotion in DEFAULT_CLASSES
    ))

    st.subheader("Model Information")

    st.write("Input Image Size: 64 × 64 RGB")
    st.write("Output Classes: 7")
    st.write("Model Type: CNN")

    st.warning(
        "This is an educational image classification system, "
        "not a psychological or medical assessment tool."
    )
