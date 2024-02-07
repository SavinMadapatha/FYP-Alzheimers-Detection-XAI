from crypt import methods
from distutils.log import debug
from flask import Flask, render_template, request
import torch
from torchvision import transforms
import PIL
from PIL import Image
from torch_utils import predict_with_confidence, load_image
import io
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/howto', methods=['GET'])
def howto():
    return render_template('howto.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        classification = None

        class_names = ['Mild Demented', 'Moderate Demented',
                       'Non Demented', 'Very Mild Demented']

        class_descriptions = {
            'Mild Demented': 'Mild cognitive decline with noticeable memory and thinking problems.',
            'Moderate Demented': 'Moderate cognitive decline, requiring assistance with daily tasks.',
            'Non Demented': 'No noticeable cognitive impairment or dementia symptoms.',
            'Very Mild Demented': 'Minimal cognitive decline; often forgetfulness but still functional.'
        }

        if request.method == 'POST':
            if 'imagefile' not in request.files:
                classification = ['-', 0, '-']
            else:
                imagefile = request.files['imagefile']
                if imagefile.filename == '':
                    classification = ['No selected file', 0, '-']
                else:
                    filename = secure_filename(imagefile.filename)
                    image_path = os.path.join('./mri_images/', filename)
                    imagefile.save(image_path)  # Save the image
                    tensor = load_image(image_path)
                    class_index, confidence = predict_with_confidence(tensor)
                    # classification = '%s (%.2f%%)' % (
                    #     class_names[class_index], confidence * 100)
                    classification = [
                        class_names[class_index], '(%.2f%%)' % (confidence * 100), class_descriptions[class_names[class_index]]]

        return render_template('predict.html', prediction=classification)

    except Exception as e:
        return render_template('predict.html', prediction=str(e))


if __name__ == '__main__':
    app.run(debug=True)
