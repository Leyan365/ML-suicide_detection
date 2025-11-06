import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import neattext.functions as nfx
import numpy as np 

MODEL_FILE = "suicide_detection_lstm_glove.h5"
TOKENIZER_FILE = 'tokenizer.pkl'
LABEL_ENCODER_FILE = "label_encoder.pkl"
MAX_SEQUENCE_LENGTH = 100 
MODEL_NAME = "LSTM+GLove"

# Load the trained artifacts
try:
    tokenizer = pickle.load(open(TOKENIZER_FILE, 'rb'))
    model = load_model(MODEL_FILE)
    lbl_target = pickle.load(open(LABEL_ENCODER_FILE, 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading required files: {e}. Please ensure '{TOKENIZER_FILE}', '{MODEL_FILE}', and '{LABEL_ENCODER_FILE}' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during file loading: {e}")
    st.stop()

# Function to clean and preprocess the input text
def preprocess_text(text, tokenizer, maxlen):
    """Cleans text, converts to sequence, and pads it."""
    # Match the cleaning steps used in training
    cleaned = nfx.remove_special_characters(text.lower())
    cleaned = nfx.remove_stopwords(cleaned)
    
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([cleaned])
    
    # Pad sequence to match training input shape
    pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    return pad

# --- MAIN STREAMLIT APPLICATION ---
if __name__ == '__main__':
    st.set_page_config(page_title="Suicidal Post Detection", layout="wide")
    st.title('Suicidal Post Detection App')
    st.subheader("Input the Post content below")
    
    sentence = st.text_area("Enter your post content here:", height=150)
    predict_btt = st.button("Predict")
    
    if predict_btt and sentence.strip() != "":
        st.write("---")
        
        # 1. Prediction Setup
        st.write(f"**Post:** *{sentence.strip()}*")

        twt_padded = preprocess_text(sentence, tokenizer, MAX_SEQUENCE_LENGTH)
        # Prediction output is a probability (float between 0 and 1)
        prediction_prob = model.predict(twt_padded, verbose=0)[0][0] 
        is_suicide = prediction_prob > 0.5

        # 2. Status Output
        if is_suicide:
            st.warning("**Potential Suicide Post**")
        else:
            st.success("**Non-Suicide Post**")

        # 3. Probability Calculation
        prob_suicide = prediction_prob * 100
        prob_non_suicide = 100 - prob_suicide
        
        # 4. Display Probabilities using Streamlit columns
        st.markdown("### Prediction Confidence")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Suicide Post Probability", value=f"{prob_suicide:.2f}%")
        
        with col2:
            st.metric(label="Non-Suicide Post Probability", value=f"{prob_non_suicide:.2f}%")
        
        st.write("---")

        # 5. Display detailed Info
        if is_suicide:
            st.info(
                f"The **{MODEL_NAME}** model predicts that there is a **{prob_suicide:.2f}%** probability "
                f"that the post content is a **Potential Suicide Post** compared to a "
                f"**{prob_non_suicide:.2f}%** probability of being a Non-Suicide Post."
            )
        else:
            st.info(
                f"The **{MODEL_NAME}** model predicts that there is a **{prob_non_suicide:.2f}%** probability "
                f"that the post content is a **Non-Suicide Post** compared to a "
                f"**{prob_suicide:.2f}%** probability of being a Potential Suicide Post."
            )
            
    elif predict_btt and sentence.strip() == "":
        st.warning("Please enter some text content to make a prediction.")