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
        'description': 'Cellulitis is a common bacterial skin infection that causes redness, swelling, and pain in the infected area of the skin.',
        'solution': 'Treatment usually involves antibiotics to eliminate the infection. It is important to keep the affected area clean and dry.'
    },
    1: {
        'label': 'Impetigo',
        'description': 'Impetigo is a highly contagious skin infection that causes red sores, often on the face, especially around the nose and mouth.',
        'solution': 'Treatment includes antibiotic ointments or oral antibiotics prescribed by a doctor. Keeping the sores clean and covered can help prevent spreading.'
    },
    2: {
        'label': 'Athlete Foot',
        'description': 'Athlete\'s foot is a fungal infection that usually begins between the toes. It commonly occurs in people whose feet have become very sweaty while confined within tight-fitting shoes.',
        'solution': 'Treatment typically involves antifungal creams or medications. Keeping feet clean and dry, and avoiding tight-fitting shoes can help prevent recurrence.'
    },
    3: {
        'label': 'Nail Fungus',
        'description': 'Nail fungus is a common condition that begins as a white or yellow spot under the tip of your fingernail or toenail. As the fungal infection goes deeper, nail fungus may cause your nail to discolor, thicken and crumble at the edge.',
        'solution': 'Treatment may include oral antifungal medications or medicated nail polish. Keeping nails trimmed and dry can help prevent further infections.'
    },
    4: {
        'label': 'Ringworm',
        'description': 'Ringworm is a common fungal infection that causes a ring-shaped rash. It is not caused by a worm, despite its name.',
        'solution': 'Treatment usually involves antifungal medications applied to the affected area. Keeping the affected area clean and dry can speed up recovery and prevent spreading.'
    },
    5: {
        'label': 'Cutaneous Larva Migrans',
        'description': 'Cutaneous larva migrans is a skin infection caused by hookworm larvae that usually results in a red, winding rash on the skin.',
        'solution': 'Treatment involves topical antiparasitic medications to kill the larvae. Avoiding contact with contaminated soil or sand can prevent infection.'
    },
    6: {
        'label': 'Chickenpox',
        'description': 'Chickenpox is an infection caused by the varicella-zoster virus. It causes an itchy rash with small, fluid-filled blisters.',
        'solution': 'Treatment focuses on relieving symptoms, such as using calamine lotion and antihistamines for itching. Vaccination can prevent chickenpox.'
    },
    7: {
        'label': 'Shingles',
        'description': 'Shingles is a viral infection that causes a painful rash. It is caused by the varicella-zoster virus, the same virus that causes chickenpox.',
        'solution': 'Treatment includes antiviral medications to reduce the severity and duration of symptoms. Pain medications and cool compresses can help relieve discomfort.'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    global img, predictions

    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Simpan file ke direktori static/uploads
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Baca dan preprocess gambar
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi penyakit
    predictions = model.predict(img)

    # Ambil label prediksi
    predicted_class = np.argmax(predictions['dense'][0])
    predicted_label = disease_info[predicted_class]['label']
    predicted_description = disease_info[predicted_class]['description']

    # Return response
    return jsonify({'label': predicted_label, 'description': predicted_description, 'filename': file.filename})



# Chatbot endpoint
@socketio.on('message')
def handle_message(message):
    print(f"Received message: {message}")

    response = process_message(message)
    emit('response', {'response': response})

def process_message(message):
    print(f"Received message: {message}")

    # Dummy response logic, implement your chatbot logic here
    if 'help me detect skin disease' in message.lower():
        print("User requests help detecting skin disease")
        return "Okay, please upload an image of the skin condition."

    # Check if the message asks for solution or treatment
    if any(keyword in message.lower() for keyword in ['solution', 'treatment', 'how to treat']):
        for disease_id, info in disease_info.items():
            if info['label'].lower() in message.lower():
                return f"Solution for {info['label']}: {info['solution']}"

    # Provide general information if no specific request found
    for disease_id, info in disease_info.items():
        if info['label'].lower() in message.lower():
            return f"{info['label']}: {info['description']}"

    return "I'm here to help! You can ask me about the symptoms or treatment of any skin disease listed above."


if __name__ == '__main__':
    socketio.run(app, debug=True)
