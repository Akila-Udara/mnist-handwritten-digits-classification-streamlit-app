
# Importing the necessary dependencies
import numpy as np
from PIL import Image
from tensorflow import keras
import streamlit as st

# Loading the pre-trained MNIST digit classification model
model = keras.models.load_model('mnist_model.h5')

# Setting the title and display an image at the top of the web app
st.title('MNIST Digit Classification App')
st.write('Upload a single image or multiple images of handwritten digits and get predictions!')

# Allowing users to upload one or more images in JPG, PNG, or JPEG formats
uploaded_images = st.file_uploader("Upload your images here", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Processing and classifying each uploaded image
if uploaded_images:
    for uploaded_image in uploaded_images:
        # Reading the uploaded image as PIL Image
        image = Image.open(uploaded_image)
        
        # Converting the image to grayscale and resizing to 28x28 pixels
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize pixel values to [0, 1]
        image_array = np.array(image) / 255.0
        
        # Reshape the image for model prediction
        processed_image = image_array.reshape(1, 28, 28)
        
        # Making predictions using the loaded model
        prediction_new_image = model.predict(processed_image)
        highest_index = np.argmax(prediction_new_image)

        # Displaying the uploaded image and the predicted label
        st.image(uploaded_image, caption='Uploaded Image.', width=200)
        st.write(f'The predicted label for the above image is: {highest_index}')
