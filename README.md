# HamletGPT-Next-Word-Prediction-with-LSTM-and-GRU

This project demonstrates the power of Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures, to predict the next word in a sequence. The model is trained on the complete text of William Shakespeare's Hamlet to learn the patterns and structure of the English language as used in this classic work.

The project includes a Streamlit web application that allows users to interact with both the trained LSTM and GRU models in real-time and compare their next-word predictions.

# Features
    LSTM and GRU Models: Two distinct models are trained to showcase and compare the performance of these advanced RNN architectures for sequence prediction.

    Data Preprocessing: The text of Hamlet is tokenized, sequenced, and padded to prepare it for model training.

    Model Training: Both models are trained with a categorical cross-entropy loss function and the Adam optimizer.

    Streamlit Application: A user-friendly web interface provides a simple way to input text and get predictions from both models.

    Model Persistence: The trained models and the tokenizer are saved, allowing for quick deployment and prediction without retraining.

# Project Structure
    next_word_prediction.py: The core script for data preprocessing, model building, training, and saving the LSTM and GRU models.

    streamlit_app.py: The Python script for the Streamlit web application.

    hamlet.txt: The raw text data file used for training.

    next_word_lstm.h5: The saved LSTM model file.

    GRU-MODEL.h5: The saved GRU model file.

tokenizer.pickle: The saved tokenizer object to ensure consistent word-to-index mapping.

How to Run Locally
Prerequisites
Python 3.7+

pip

Installation
Clone the repository:

Bash

git clone https://github.com/kirantushar10/HamletGPT-Next-Word-Prediction-with-LSTM-and-GRU.git
cd HamletGPT-Next-Word-Prediction-with-LSTM-and-GRU
Install the required libraries:

Bash

pip install -r requirements.txt
Note: You'll need to create a requirements.txt file with the following dependencies:

tensorflow
nltk
streamlit
numpy
pandas
scikit-learn
Run the main script to train the models and save the files:

Bash

python next_word_prediction.py
This will create the .h5 model files and the tokenizer.pickle file.

Launch the Streamlit application:

Bash

streamlit run streamlit_app.py
Your default web browser should open to the application. If not, open http://localhost:8501.

Live Demo
You can test the application live at: https://shorturl.at/nQz2Z

Model Architectures
LSTM Model
The LSTM model is designed with two layers to capture complex dependencies in the text.

Embedding Layer: Converts word indices into dense vectors.

LSTM Layers: Two LSTM layers with 150 and 100 units, respectively, to learn long-range dependencies.

Dropout Layer: A dropout rate of 0.2 is applied to prevent overfitting.

Dense Output Layer: A final dense layer with a softmax activation function outputs probabilities for each word in the vocabulary.

GRU Model
The GRU model offers a slightly simpler, and often more computationally efficient, alternative to the LSTM.

Embedding Layer: Similar to the LSTM model.

GRU Layers: Two GRU layers with 150 and 100 units.

Dropout Layer: A dropout rate of 0.2 is applied for regularization.

Dense Output Layer: A final dense layer with a softmax activation function.

How it Works
Data Loading: The gutenberg corpus from NLTK is used to load the text of Hamlet.

Tokenization: The text is converted into a sequence of numbers, where each number represents a unique word.

Sequence Generation: The text is split into n-gram sequences. For each line, every possible sequence of words is created and padded to a uniform length. For example, from "To be or not to be," sequences like "To be," "To be or," and "To be or not" are created.

Model Training: The model is trained on these sequences, learning to predict the last word of a sequence given the preceding words.

Prediction: When a new phrase is provided, it is tokenized, padded to the correct length, and fed into the model. The model outputs a probability distribution over the entire vocabulary, and the word with the highest probability is selected as the prediction.
