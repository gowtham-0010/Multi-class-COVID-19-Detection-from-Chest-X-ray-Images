import streamlit as st
import tensorflow as tf
from keras.layers import Dense
from PIL import Image
import numpy as np

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    layout="centered"
)

# ---------------- KERAS PATCH ----------------
# Removes the incompatible "quantization_config" field
# from Dense layer configs while loading the model.

_original_dense_from_config = Dense.from_config

@classmethod
def patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_dense_from_config(config)

Dense.from_config = patched_dense_from_config

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "covid_classifier_model.h5",
        compile=False
    )
    return model

model = load_model()

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)

    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array / 255.0

# ---------------- UI ----------------
st.title("🩻 Chest X-Ray Classifier")
st.markdown(
    """
    Upload a chest X-ray image and the model will classify it as:

    - COVID-19
    - Normal
    - Viral Pneumonia
    """
)

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded X-Ray",
        use_container_width=True
    )

    st.write("Classifying...")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    class_names = [
        "Covid",
        "Normal",
        "Viral Pneumonia"
    ]

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")