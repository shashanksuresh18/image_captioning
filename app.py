import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json

# Load model and tokenizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")

# Load the trained captioning model
model = load_model(MODEL_PATH)

# Load the correct tokenizer
with open(TOKENIZER_PATH, 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))  # Convert dict to JSON string

# Load VGG16 for feature extraction (include top layers to get 4096-dimensional features)
vgg_model = VGG16(weights="imagenet")
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)  # Use fc2 layer (4096-dim)

# Parameters
vocab_size = len(tokenizer.word_index) + 1  # Ensure this matches training
max_length = 46  # Ensure this matches training

# Debugging
print(f"Tokenizer vocabulary size: {vocab_size}")
print(f"Expected max length: {max_length}")

# Helper functions
def idx_to_word(integer, tokenizer):
    """Convert an index to a word using the tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(image_path, vgg_model):
    """Extract features from the image using VGG16."""
    image = load_img(image_path, target_size=(224, 224))  # Resize the image
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess for VGG16
    features = vgg_model.predict(image, verbose=0)  # Extract features
    print(f"Extracted features shape: {features.shape}")
    return features

def predict_caption(model, features, tokenizer, max_length):
    """Generate a caption for the given image features."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)  # Ensure sequence matches model's expectations
        print(f"Padded sequence shape: {sequence.shape}")
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Extract features using VGG16
            try:
                features = extract_features(filepath, vgg_model)
                
                # Generate caption
                caption = predict_caption(model, features, tokenizer, max_length)
                
                return render_template('index.html', caption=caption, image_path=file.filename)
            except Exception as e:
                print(f"Error processing the image: {e}")
                return "Error processing the image. Please try again.", 500
    
    return render_template('index.html')

# Start the Flask app
if __name__ == '__main__':
    app.run()
