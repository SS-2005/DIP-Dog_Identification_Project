import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Add CORS support
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model with custom objects for TensorFlow Hub layer
try:
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model('full-image-set-mobilenetv2-adam.h5', 
                                      custom_objects=custom_objects)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load labels
try:
    labels_df = pd.read_csv('labels.csv')
    unique_breeds = np.sort(labels_df.breed.unique())
    print("Labels loaded successfully")
except:
    # If labels.csv doesn't exist, create a placeholder
    unique_breeds = [f'breed_{i}' for i in range(120)]
    print("Warning: labels.csv not found. Using placeholder breed names.")

# Image size for model input
IMG_SIZE = 224

def preprocess_image(img):
    """
    Preprocess image for model prediction
    """
    # Convert to array and resize
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    # Expand dimensions to create batch of size 1
    img_array = tf.expand_dims(img_array, axis=0)
    # Normalize the image (same as training)
    img_array = img_array / 255.
    return img_array

def predict_breed(img):
    """
    Predict dog breed from image
    """
    if model is None:
        raise Exception("Model not loaded")
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img)
    
    # Get top prediction
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_breed = unique_breeds[predicted_idx]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_breeds = unique_breeds[top_5_indices]
    top_5_confidences = predictions[0][top_5_indices]
    
    top_predictions = []
    for i in range(5):
        top_predictions.append({
            'breed': top_5_breeds[i],
            'confidence': float(top_5_confidences[i] * 100)
        })
    
    return predicted_breed, confidence * 100, top_predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files and 'url' not in request.form:
            return jsonify({'error': 'No image provided'})
        
        if 'file' in request.files:
            # Get the file from post request
            file = request.files['file']
            
            # Check if file is selected
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            
            # Read the image
            img = Image.open(file.stream)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
        elif 'url' in request.form and request.form['url']:
            # Get image from URL
            image_url = request.form['url']
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        # Make prediction
        breed, confidence, top_predictions = predict_breed(img)
        
        # Convert image to base64 for display
        buffered = BytesIO()
        # Resize image for display while maintaining aspect ratio
        img.thumbnail((300, 300))
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'breed': breed,
            'confidence': round(confidence, 2),
            'top_predictions': top_predictions,
            'image': img_str
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/random_dog')
def random_dog():
    try:
        # Get random dog image from Dog API
        response = requests.get('https://dog.ceo/api/breeds/image/random')
        image_url = response.json()['message']
        
        return jsonify({'url': image_url})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)