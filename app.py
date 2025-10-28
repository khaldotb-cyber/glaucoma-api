from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load your glaucoma model
print("Loading glaucoma detection model...")
try:
    model = tf.keras.models.load_model('final_glaucoma_detection_model.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def predict_glaucoma(image_bytes):
    """Make prediction on image"""
    try:
        if model is None:
            return {'status': 'error', 'error': 'Model not loaded'}
        
        # Preprocess image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        glaucoma_prob = float(prediction)
        
        if glaucoma_prob > 0.5:
            result_class = 'Glaucoma'
            confidence = glaucoma_prob
        else:
            result_class = 'Healthy'
            confidence = 1 - glaucoma_prob
            
        return {
            'prediction': result_class,
            'confidence': float(confidence),
            'glaucoma_probability': glaucoma_prob,
            'healthy_probability': 1 - glaucoma_prob,
            'is_glaucoma': result_class == 'Glaucoma',
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

@app.route('/')
def home():
    return jsonify({
        'status': 'Glaucoma Detection API', 
        'version': '1.0',
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'error': 'No image file provided'})
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return jsonify({'status': 'error', 'error': 'Empty image file'})
        
        print(f"📸 Processing image: {len(image_bytes)} bytes")
        
        # Make prediction
        result = predict_glaucoma(image_bytes)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
