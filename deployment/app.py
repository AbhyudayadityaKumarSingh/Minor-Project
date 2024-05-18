from __future__ import division, print_function
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = r'C:\Users\abhyu\Downloads\Eye-Disease-Classigication-and-Detection-main\Eye-Disease-Classigication-and-Detection-main\final_model.h5'
model = load_model(MODEL_PATH)

img_height, img_width = 224, 224
class_labels = ['Cataract', 'Glaucoma', 'Diabetic Retinopathy', 'Normal']

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return predictions

def custom_decode_predictions(predictions, class_labels):
    decoded_predictions = []
    for sample_probs in predictions:
        decoded_sample = []
        for class_prob, label in zip(sample_probs, class_labels):
            decoded_sample.append((label, float(class_prob)))
        decoded_predictions.append(decoded_sample)
    return decoded_predictions


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = 'temp_image.jpg'
        file.save(file_path)
        predictions = model_predict(file_path, model)
        decoded_predictions = custom_decode_predictions(predictions, class_labels)
        
        # Finding the element with the highest probability
        max_probability = 0
        max_label = ""
        for label, probability in decoded_predictions[0]:
            if probability > max_probability:
                max_probability = probability
                max_label = label

        print(decoded_predictions[0])
        
        response = {'label': max_label, 'probability': max_probability}
        return jsonify(response)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)