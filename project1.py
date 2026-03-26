from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = load_model('glaucoma_model.h5')
print("Model loaded successfully")

THRESHOLD = 0.6

def preprocess(img):
    img = img.resize((128, 128))  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_img = preprocess(img)
        prediction = model.predict(processed_img)[0][0]

        result = {
            'prediction': 'Glaucoma Detected' if prediction > THRESHOLD else 'No Glaucoma Detected',
            'confidence': float(prediction if prediction > THRESHOLD else 1 - prediction)
        }
        print("Sending result:", result)
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to process image or predict'}), 500

if __name__ == "__main__":
    app.run(debug=True)
