import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

#Load pickle file of trained model
model = tf.keras.models.load_model("letter_model.h5")

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 28x28 pixels
    resized = cv2.resize(gray, (28, 28))
    # Normalize the pixel values to be between 0 and 1
    normalized = resized / 255.0
    # Reshape the image to match the input shape of the model (1 channel)
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    return reshaped

def main():
    # Set app title and description
    st.title("Handwritten Alphabet Recognition")
    st.write("Upload an image of a handwritten alphabet and let the model predict the letter.")

    # Create file uploader
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image from the file uploader
        image = Image.open(uploaded_file)
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions using the pre-trained model
        predictions = model.predict(processed_image)
        # Get the predicted letter
        predicted_letter = chr(np.argmax(predictions) + 65)

        # Display the uploaded image and the predicted letter
        st.image(image, use_column_width=True)
        st.write("Predicted Letter:", predicted_letter)

if __name__ == '__main__':
    main()
