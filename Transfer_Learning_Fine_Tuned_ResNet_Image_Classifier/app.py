import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title = "ResNet Classifier", layout = "centered")

import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, TFResNetModel
from config import IMAGE_SIZE, MODEL_NAME, NUM_CLASSES

# ==========================
# Build Model Architecture
# ==========================
@st.cache_resource
def load_model():
    base_model = TFResNetModel.from_pretrained(MODEL_NAME)
    base_model.trainable = False
    input_layer = tf.keras.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = "input_image")
    x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]))(input_layer)
    x = base_model(pixel_values = x, training=False).pooler_output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation = "relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation = "softmax")(x)

    model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
    model.load_weights("saved_model/resnet_weights.h5")
    return model

# Loading model and processor
model = load_model()
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# ==========================
# Streamlit UI
# ==========================
st.title("Transfer Learning & Fine-Tuned ResNet Image Classifier")
st.markdown("Upload an image and classify it using your custom-trained ResNet model.")

uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption = "Uploaded Image", use_column_width = True)

    # Preprocessing the input image
    img_array = np.array(image)
    pixel_values = processor(images=img_array, return_tensors = "np")["pixel_values"][0]
    pixel_values = np.transpose(pixel_values, (1, 2, 0))     # Convert CHW to HWC
    pixel_values = np.expand_dims(pixel_values, axis = 0)     # Add batch dim

    # Prediction
    predictions = model.predict(pixel_values)[0]
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    CLASS_NAME = ['class_1', 'class_2', 'class_3', 'class_4']
    
    # Displaying Result
    #class_name = f"class{predicted_index + 1}"  # Replace with actual class names if needed
    class_name = CLASS_NAME[predicted_index]
    st.markdown(f"Predicted Class: `{class_name}`")
    st.markdown(f"Confidence Score: `{confidence:.2f}%`")