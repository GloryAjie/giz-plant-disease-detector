# jsonify is used to turn an html to a json file
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# The OS module in Python provides functions for creating and removing a directory (folder), fetching its contents, changing and identifying the current directory
import os

# Pil or (flow) is used to import images
from PIL import Image as PILImage

# io module allows us to manage the file-related input and output operations.
import io

# set the environment variable to disable oneDNN Optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Instantiate Flask before you can use it
app = Flask(__name__)

# Load the training CNN Model
MODEL_PATH =  'NEW_H5/A_plant_disease_model.h5'
model = load_model(MODEL_PATH)

# Compile the model to avoid any error
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Tell your models the names of the plant diseases
disease_classes = ['Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple',
                   'Blueberry Healthy', 'Cherry Powdery Mildew', 'Healthy Cherry',
                   'Corn Cercospora Leaf Spot Gray Leaf Spot', 'Corn Common Rust',
                   'Corn Northern Leaf Blight', 'Healthy Corn', 'Grape Black Rot',
                   'Grape Esca (Black Measles)', 'Grape Leaf Blight', 'Healthy Grape',
                   'Citrus Greening', 'Peach Bacterial Spot', 'Healthy Peach',
                   'Pepper Bell Bacterial Spot', 'Healthy Pepper Bell', 'Potato Early Blight',
                   'Potato Late Blight', 'Healthy Potato', 'Raspberry Healthy',
                   'Soybean Healthy', 'Squash Powdery Mildew', 'Strawberry Leaf Scorch',
                   'Healthy Strawberry', 'Tomato Bacterial Spot', 'Tomato Early Blight',
                   'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
                   'Tomato Spider Mites Two-Spotted Spider Mite', 'Tomato Target Spot',
                   'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Healthy Tomato']

# Using Flask on your html file; the '@' is to refer to the variable 'app' where Flask has been assigned
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods = ['POST'])
def predict():

    # check if the file is in the request.
    if 'file' not in request.files:
        return jsonify({'error': 'No File Provided/Uploaded'}), 400
    
    # if file is in the request, get file and process it. Also convert your image files to RGB
    img_file = request.files['file']
    img= PILImage.open(img_file.stream).convert('RGB')

    # Resize your image
    img= img.resize((128,128))

    # Make your images in an array so that ML can happen better
    img_array=image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 128.0

    # Now, arrange for the prediction
    try:
        prediction = model.predict(img_array)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    class_index = np.argmax(prediction, axis = 1)[0]
    confidence = np.max(prediction)*100

    # Get your prediction to show on the page
    predicted_disease = disease_classes[class_index]

    return jsonify({'predicted_disease': predicted_disease, 'confidence': confidence})
    
    # Close your Flask. Remember to always close your flask.
    if __name__ =='__main__':
        app.run(debug=True)
        