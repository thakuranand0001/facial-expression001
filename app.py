
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

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Face Image",
        use_container_width=True
    )

    # Predict button
    if st.button("Predict Expression", type="primary"):

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

                # Preprocess image
                resized_image = image.resize(IMAGE_SIZE)

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # If training used a Rescaling layer,
                # do not divide by 255 again.
                # Otherwise, use:
                # image_array = image_array / 255.0

                # Model prediction
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                if len(predictions) != len(classes):
                    st.error("Model classes do not match class labels.")

                else:
                    predicted_index = np.argmax(predictions)
                    predicted_expression = classes[predicted_index]

                    # Display result
                    st.success("Prediction Completed!")

                    st.subheader("Predicted Expression")

                    st.markdown(
                        f"# {EMOJI.get(predicted_expression, '🙂')} "
                        f"{predicted_expression.title()}"
                    )

                    # Display scores
                    st.subheader("Expression Scores")

                    for emotion, score in sorted(
                        zip(classes, predictions),
                        key=lambda x: x[1],
                        reverse=True
                    ):
                        st.write(
                            f"{EMOJI.get(emotion, '•')} "
                            f"{emotion.title()}: 
# Model prediction
predictions = model.predict(image_array, verbose=0)[0]

predicted_index = int(np.argmax(predictions))
predicted_expression = classes[predicted_index]

# Prediction score
prediction_score = float(predictions[predicted_index]) * 100

st.success("Prediction Completed!")

st.subheader("Prediction Result")

st.markdown(
    f"# {EMOJI.get(predicted_expression, '🙂')} "
    f"{predicted_expression.title()}"
)

st.metric(
    "Prediction Score",
    f"{prediction_score:.2f}%"
)

st.progress(
    float(np.clip(prediction_score / 100, 0, 1))
)

st.write("All Expression Scores")

for emotion, score in sorted(
    zip(classes, predictions),
    key=lambda x: x[1],
    reverse=True
):
    st.write(f"{emotion.title()}: {score * 100:.2f}%")
    st.progress(float(np.clip(score, 0, 1)))

                        )

                        st.progress(
                            float(np.clip(score, 0, 1))
                        )

            except Exception as e:
                st.error(f"Prediction Error: {e}")

else:
    st.info("Please upload a face image to start prediction.")

st.caption(
    "Educational demonstration only. Facial expressions "
    "do not reliably reveal a person's internal emotions."
)
