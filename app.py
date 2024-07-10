import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
from flask_socketio import SocketIO, emit
from keras.layers import TFSMLayer

app = Flask(__name__)
socketio = SocketIO(app)

# Path to the SavedModel directory
saved_model_dir = 'model/skin_disease_model_saved'

# Load model using TFSMLayer
model = tf.keras.Sequential([
    TFSMLayer(saved_model_dir, call_endpoint='serving_default')
])

# Disease labels and detailed information
disease_info = {
    0: {
        'label': 'Cellulitis',
        'description': 'Cellulitis is a common bacterial skin infection that causes redness, swelling, and pain in the infected area of the skin.'
    },
    1: {
        'label': 'Impetigo',
        'description': 'Impetigo is a highly contagious skin infection that causes red sores, often on the face, especially around the nose and mouth.'
    },
    2: {
        'label': 'Athlete Foot',
        'description': 'Athlete\'s foot is a fungal infection that usually begins between the toes. It commonly occurs in people whose feet have become very sweaty while confined within tight-fitting shoes.'
    },
    3: {
        'label': 'Nail Fungus',
        'description': 'Nail fungus is a common condition that begins as a white or yellow spot under the tip of your fingernail or toenail. As the fungal infection goes deeper, nail fungus may cause your nail to discolor, thicken and crumble at the edge.'
    },
    4: {
        'label': 'Ringworm',
        'description': 'Ringworm is a common fungal infection that causes a ring-shaped rash. It is not caused by a worm, despite its name.'
    },
    5: {
        'label': 'Cutaneous Larva Migrans',
        'description': 'Cutaneous larva migrans is a skin infection caused by hookworm larvae that usually results in a red, winding rash on the skin.'
    },
    6: {
        'label': 'Chickenpox',
        'description': 'Chickenpox is an infection caused by the varicella-zoster virus. It causes an itchy rash with small, fluid-filled blisters.'
    },
    7: {
        'label': 'Shingles',
        'description': 'Shingles is a viral infection that causes a painful rash. It is caused by the varicella-zoster virus, the same virus that causes chickenpox.'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the file to the static/uploads directory
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)
    
    # Read and preprocess the image
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict the disease
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = disease_info[predicted_class]['label']
    predicted_description = disease_info[predicted_class]['description']
    
    return jsonify({'label': predicted_label, 'description': predicted_description, 'filename': file.filename})

# Chatbot endpoint
@socketio.on('message')
def handle_message(message):
    response = process_message(message)
    emit('response', {'response': response})

def process_message(message):
    # Dummy response logic, implement your chatbot logic here
    if 'help me detect skin disease' in message.lower():
        return "Okay, please upload an image of the skin condition."
    
    for disease_id, info in disease_info.items():
        if info['label'].lower() in message.lower():
            return info['description']
    
    return "I'm here to help! You can ask me about the symptoms or treatment of any skin disease listed above."

if __name__ == '__main__':
    socketio.run(app, debug=True)
