import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os

# Load models and tokenizer
WORKING_DIR = r'C:\Users\Shekhani Laptops\latest'
model = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'best_model.keras'))
vgg_model = VGG16()
vgg_model = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

with open(os.path.join(WORKING_DIR, 'tokenizer.pickle'), 'rb') as f:
    tokenizer = pickle.load(f)

with open(os.path.join(WORKING_DIR, 'max_length.pkl'), 'rb') as f:
    max_length = pickle.load(f)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def main():
    st.title("Image Caption Generator")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        feature = vgg_model.predict(image, verbose=0)
        caption = predict_caption(model, feature, tokenizer, max_length)
        
        st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
        st.write(caption)

if __name__ == "__main__":
    main()
