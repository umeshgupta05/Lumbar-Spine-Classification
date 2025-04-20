from flask import Flask, render_template, request
import json
import pydicom
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load model
model = load_model('efficientnetb0_model.keras')

# Condition labels
labels = {
    0: "Spinal Canal Stenosis",
    1: "Left Neural Foraminal Narrowing",
    2: "Right Neural Foraminal Narrowing",
    3: "Left Subarticular Stenosis",
    4: "Right Subarticular Stenosis"
}

# Load symptoms and precautions
with open("conditions.json") as f:
    info_dict = json.load(f)

# DICOM image preprocessing
def preprocess_dcm(file):
    dcm = pydicom.dcmread(file)
    img = dcm.pixel_array.astype(np.float32)
    img = np.stack((img,) * 3, axis=-1)
    img = tf.image.resize(img, (224, 224)).numpy()
    img /= 255.0
    return np.expand_dims(img, axis=0)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files['file']
    image = preprocess_dcm(file)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction[0])
    predicted_label = labels[predicted_class]

    symptoms = info_dict.get(predicted_label, {}).get('symptoms', [])
    precautions = info_dict.get(predicted_label, {}).get('precautions', [])

    # Save the uploaded DICOM image as PNG
    file.seek(0)  # Reset file pointer
    dcm = pydicom.dcmread(file)
    img_array = dcm.pixel_array
    img_filename = f"{uuid.uuid4().hex}.png"
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    plt.imsave(img_path, img_array, cmap='gray')

    return render_template("index.html",
                           prediction=predicted_label,
                           symptoms=symptoms,
                           precautions=precautions,
                           image_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
