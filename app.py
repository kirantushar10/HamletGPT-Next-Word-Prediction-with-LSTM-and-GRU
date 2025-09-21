from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding,LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st

import nltk
import pandas as pd

## LOAD LSTM MODEL
model = load_model('next_word_lstm.h5')
model2 = load_model('GRU-MODEL.h5')

## LOAD THE TOKENIZER
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


## FUNCTION TO PREDICT THE NEXT WORD
def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len-1):]
    token_list = pad_sequences([token_list],maxlen = max_seq_len-1, padding='pre')
    predicted = model.predict(token_list, verbose = 0)
    predicted_word_index = np.argmax(predicted, axis = 1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

## STREAMLIT APP
st.title("Next Word Prediction with LSTM-GRU")
input_text = st.text_input('Enter the sequence of words',"To be or not to")
if st.button('Predict Next Word'):
    max_seq_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer, input_text, max_seq_len)
    next_word_2 = predict_next_word(model2,tokenizer, input_text, max_seq_len)
    st.write(f"Next Word (LSTM): {next_word}")
    st.write(f"Next Word (GRU) : {next_word_2}")
