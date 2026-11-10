
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
