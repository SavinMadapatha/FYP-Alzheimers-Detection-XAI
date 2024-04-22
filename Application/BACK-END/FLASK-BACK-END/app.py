from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from model import (ensemble_model, preprocess_image, predict_with_confidence, 
                   apply_gradcam_AlexNet, apply_gradcam_efficientnet, 
                   generate_lime_and_highlighted, generate_saliency_map)
from flask_cors import CORS, cross_origin

app = Flask(__name__, static_folder='mri_images')
CORS(app)

# Assuming predefined class names and descriptions
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

class_descriptions = {
    'Mild Demented': 'Mild cognitive decline with noticeable memory and thinking problems.',
    'Moderate Demented': 'Moderate cognitive decline, requiring assistance with daily tasks.',
    'Non Demented': 'No noticeable cognitive impairment or dementia symptoms.',
    'Very Mild Demented': 'Minimal cognitive decline; often forgetfulness but still functional.'
}

# dummy user data
users = {
    "admin": "admin"
}

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    """
    here typically the connected DB should be queries, since this is not the highest concern of the research, login (for securty) has been simulated, which
    can be implemented if this goes into the production stage.
    """
    if username in users and check_password_hash(users[username], password):
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"message": "Invalid username or password"}), 401

@app.route('/mri_images/<filename>')
@cross_origin()
def serve_mri_image(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if 'imagefile' not in request.files:
        return jsonify(['-', '0', '-', ''])
    
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return jsonify(['No selected file', '0', '-', ''])
    
    try:
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.static_folder, filename)
        imagefile.save(image_path)
        
        img_tensor = preprocess_image(image_path)
        class_index, confidence = predict_with_confidence(img_tensor)

        classification = [
            class_names[class_index], 
            f'{confidence * 100:.2f}%', 
            class_descriptions[class_names[class_index]]
        ]
        return jsonify(classification)
    except Exception as e:
        print(f"Exception: {e}")
        return jsonify(['Error processing the image', '0', '-', ''])

@app.route('/generate_interpretations', methods=['POST'])
def generate_interpretations():
    if 'imagefile' not in request.files:
        return jsonify(['No image file provided'])

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return jsonify(['No selected file'])

    try:
        # Save the image to a temporary path or process in memory
        image_path = os.path.join(app.static_folder, secure_filename(imagefile.filename))
        imagefile.save(image_path)
        img_tensor = preprocess_image(image_path)

        # Common URL array for all XAI technique images
        xai_image_urls = []

        # Generate Grad-CAM images and append their URLs
        xai_image_urls.append(apply_gradcam_AlexNet(ensemble_model.model1, img_tensor, ensemble_model.model1.conv6, image_path))
        xai_image_urls.append(apply_gradcam_efficientnet(ensemble_model.model2, img_tensor, "SEBlock", image_path))

        # Generate LIME explanation images and apend their URLs
        alex_lime, alex_highlight_lime = generate_lime_and_highlighted(image_path, ensemble_model.model1, "AlexNet")
        efficient_lime, efficient_highlight_lime = generate_lime_and_highlighted(image_path, ensemble_model.model2, "EfficientNet")
        # xai_image_urls.append(generate_lime_explanation(image_path, ensemble_model.model1))
        # xai_image_urls.append(generate_lime_explanation(image_path, ensemble_model.model2))

        xai_image_urls.append(alex_lime)
        xai_image_urls.append(efficient_lime)
        xai_image_urls.append(alex_highlight_lime)
        xai_image_urls.append(efficient_highlight_lime)

        xai_image_urls.append(generate_saliency_map(ensemble_model.model1, img_tensor, image_path, 'AlexNet'))
        xai_image_urls.append(generate_saliency_map(ensemble_model.model2, img_tensor, image_path, 'EfficientNet'))
        
        # xai_image_urls.append(test_lime(ensemble_model.model1, image_path, 'AlexNet'))
        # xai_image_urls.append(test_lime(ensemble_model.model2, image_path, 'EfficientNet'))

        return jsonify(xai_image_urls)
    except Exception as e:
        print(f"Exception: {e}")
        return jsonify(['Error generating interpretations'])


if __name__ == '__main__':
    app.run(debug=True)
