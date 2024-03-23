from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from model import ensemble_model, load_image, predict_with_confidence
from flask_cors import CORS

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Assuming you have predefined class names and descriptions
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
class_descriptions = {
    'Mild Demented': 'Mild cognitive decline with noticeable memory and thinking problems.',
    'Moderate Demented': 'Moderate cognitive decline, requiring assistance with daily tasks.',
    'Non Demented': 'No noticeable cognitive impairment or dementia symptoms.',
    'Very Mild Demented': 'Minimal cognitive decline; often forgetfulness but still functional.'
}

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route called")
    if 'imagefile' not in request.files:
        print("No file part")
        return jsonify(['-', '0', '-'])

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        print("No selected file")
        return jsonify(['No selected file', '0', '-'])

    try:
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join('./mri_images/', filename)
        print(f"Saving image to {image_path}")
        imagefile.save(image_path)
        
        print("Loading and predicting image")
        tensor = load_image(image_path)
        class_index, confidence = predict_with_confidence(tensor)
        
        classification = [class_names[class_index], f'{confidence * 100:.2f}%', class_descriptions[class_names[class_index]]]
        print(f"Classification: {classification}")
        return jsonify(classification)
    except Exception as e:
        print(f"Exception during processing: {e}")
        return jsonify(['Error processing the image', '0', '-'])

if __name__ == '__main__':
    app.run(debug=True)
