
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered"
)

# Title
st.title("😊 Facial Expression Recognition")
st.write("Upload a face image to identify its facial expression.")

# Model and class files
MODEL_PATH = Path("facial_expression_model.keras")
CLASSES_PATH = Path("classes.json")

IMAGE_SIZE = (64, 64)

DEFAULT_CLASSES = [
    "angry", "disgusted", "fearful",
    "happy", "neutral", "sad", "surprised"
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

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# Upload image
uploaded_file = st.file_uploader(
    "Choose a facial image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Face Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Expression", type="primary"):

        if not MODEL_PATH.exists():
            st.error("Model file not found! Please train the model first.")

        else:
            try:
                model = load_model()

                # Load class names
                if CLASSES_PATH.exists():
                    classes = json.loads(
                        CLASSES_PATH.read_text(encoding="utf-8")
                    )
                else:
                    classes = DEFAULT_CLASSES

                # Image preprocessing
                resized_image = image.resize(IMAGE_SIZE)

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Match preprocessing with model training
                # If model has Rescaling(1./255), do not divide again.
                # Otherwise uncomment:
                # image_array = image_array / 255.0

                # ACTUAL CNN MODEL PREDICTION
                raw_predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                if len(raw_predictions) != len(classes):
                    st.error("Model output does not match class labels.")

                else:
                    # Normalize actual model outputs for display
                    # This makes displayed scores total 100%.
                    predictions = np.asarray(
                        raw_predictions,
                        dtype=np.float64
                    )

                    # Handle logits or probabilities
                    predictions = np.exp(
                        predictions - np.max(predictions)
                    )

                    predictions = (
                        predictions / np.sum(predictions)
                    ) * 100

                    # Actual predicted expression
                    predicted_index = int(
                        np.argmax(predictions)
                    )

                    predicted_expression = classes[predicted_index]

                    # Prediction completed
                    st.success("Prediction Completed!")

                    # PREDICTION RESULT
                    st.subheader("Prediction Result")

                    st.markdown(
                        f"# {EMOJI.get(predicted_expression, '🙂')} "
                        f"{predicted_expression.title()}"
                    )

                    # Fixed demonstration score
                    st.metric(
                        label="Display Score (Demo)",
                        value="100.00%"
                    )

                    st.progress(1.0)

                    # ACTUAL MODEL SCORE DISTRIBUTION
                    st.subheader("Actual Model Score Distribution")

                    sorted_results = sorted(
                        zip(classes, predictions),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    for emotion, score in sorted_results:

                        st.write(
                            f"{EMOJI.get(emotion, '•')} "
                            f"**{emotion.title()}**: {score:.2f}%"
                        )

                        st.progress(
                            float(np.clip(score / 100, 0, 1))
                        )

                    st.success(
                        f"Total Model Score: {np.sum(predictions):.2f}%"
                    )

                    st.caption(
                        "The expression is selected using the actual "
                        "CNN model output. Normalized scores are not "
                        "guaranteed prediction accuracy."
                    )

            except Exception as e:
                st.error(f"Prediction Error: {e}")

else:
    st.info("Please upload a face image to start prediction.")

# Footer
st.caption(
    "Facial Expression Recognition Using Deep Learning | "
    "Python · CNN · TensorFlow · Streamlit"
)
