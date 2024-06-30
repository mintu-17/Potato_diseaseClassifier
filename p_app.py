import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Determine the latest model version
model_directory = "../models"
model_files = [f for f in os.listdir(model_directory) if f.endswith('.keras')]
latest_model_file = max(model_files, key=lambda f: int(f.split('.')[0]))

# Load the trained model
MODEL_PATH = os.path.join(model_directory, latest_model_file)
model = load_model(MODEL_PATH)

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))  # Adjust target size if necessary
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image
    return img

# Define the class names based on your model's training
class_names = ['Healthy', 'Early Blight', 'Late Blight']

# Streamlit app
st.title("Potato Leaf Disease Classifier")
st.write("Upload a potato leaf image to classify it as healthy, early blight, or late blight.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = preprocess_image(uploaded_file)

    # Make prediction
    predictions = model.predict(image)
    pred_class_index = np.argmax(predictions)
    pred_class = class_names[pred_class_index]
    confidence = predictions[0][pred_class_index]

    st.write(f"Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}")
