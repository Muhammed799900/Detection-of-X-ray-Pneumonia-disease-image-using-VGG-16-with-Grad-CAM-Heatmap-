import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
from gradcam import get_gradcam, apply_heatmap

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['MODEL_PATH'] = 'model.h5'

# Load the VGG-16 model
model = tf.keras.models.load_model(app.config['MODEL_PATH'])

# Class names
class_names = ['Normal', 'Pneumonia']

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img_array = preprocess_image(filepath)

    # Predict
    predictions = model.predict(img_array)[0]
    confidence = round(np.max(predictions) * 100, 2)
    predicted_class = class_names[np.argmax(predictions)]

    # Apply Grad-CAM
    heatmap = get_gradcam(model, img_array, 'block5_conv3')
    gradcam_img = apply_heatmap(filepath, heatmap)

    # Save Grad-CAM output
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_' + filename)
    cv2.imwrite(gradcam_path, gradcam_img)

    return render_template('result.html', prediction=predicted_class, confidence=confidence, gradcam_path=gradcam_path)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

