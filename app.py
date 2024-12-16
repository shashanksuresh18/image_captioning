import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json

# Load model and tokenizer
MODEL_PATH = "best_model.keras"
TOKENIZER_PATH = "tokenizer.json"

# Load the trained captioning model
model = load_model(MODEL_PATH)

# Load the correct tokenizer
with open(TOKENIZER_PATH, 'r') as f:
    tokenizer = tokenizer_from_json(f.read())

# Load VGG16 for feature extraction
vgg_model = VGG16(weights="imagenet")
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Parameters
vocab_size = len(tokenizer.word_index) + 1
max_length = 46

# Helper functions
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(image_path, vgg_model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = vgg_model.predict(image, verbose=0)
    return features

def predict_caption(model, features, tokenizer, max_length):
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

# Streamlit app
st.title("Image Caption Generator")
st.write("Upload an image and generate a caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image
    image_path = os.path.join("uploaded_image.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the image
    st.image(image_path, caption="Uploaded Image.", use_column_width=True)

    # Extract features and predict caption
    features = extract_features(image_path, vgg_model)
    caption = predict_caption(model, features, tokenizer, max_length)

    st.write("**Generated Caption:**")
    st.write(caption)
