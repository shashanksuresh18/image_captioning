import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json

# ------------------------ Model & Tokenizer Initialization ------------------------

# Load model and tokenizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")

# Load the trained captioning model
model = load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))  # Convert dict to JSON string

# Load VGG16 for feature extraction
vgg_model = VGG16(weights="imagenet")
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)  # fc2 layer for 4096-dim features

# Parameters
vocab_size = len(tokenizer.word_index) + 1
max_length = 46

# ------------------------ Helper Functions ------------------------

def idx_to_word(integer, tokenizer):
    """Convert an index to a word using the tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(image):
    """Extract features from the image using VGG16."""
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = vgg_model.predict(image, verbose=0)
    return features

def predict_caption(features):
    """Generate caption for the given image features."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# ------------------------ Streamlit UI Code ------------------------

# Page Config
st.set_page_config(page_title="Image Caption Generator", page_icon="📷", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #f0f4f8, #e6e6e6);
            margin: 0;
        }
        .stApp {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            color: black;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .stTitle {
            text-shadow: 1px 1px 2px #000000;
        }
    </style>
    """, unsafe_allow_html=True)

# Page Header
st.title("📷 Image Caption Generator")
st.write("Upload an image and generate a descriptive caption using a trained AI model.")

# Layout
col1, col2 = st.columns(2)

# File Upload Section
with col1:
    uploaded_file = st.file_uploader("**Choose an Image**", type=["jpg", "jpeg", "png"])

# Image Display & Caption Prediction
if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file)
    
    # Display Uploaded Image
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate Caption
    with col2:
        st.subheader("Generated Caption:")
        features = extract_features(image)
        caption = predict_caption(features)
        st.success(f"**{caption}**")

# Footer
st.markdown("""
    ---
    Developed with ❤️ by **Evoastra team**
""")
