import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import keras.backend as K

class RNN_Model:
    """
    This is a class representing a RNN Model.
    It is capable of doing binary classification, specifically on texts
    Following methods are defined:
    - encode_words: encode words into integers
    - create_model: create a keras model
    - train_model: train the model
    - evaluate_model: evaluate the model
    - predict_model: predict the model
    - get_binary_predictions: get the binary predictions
    """

    def __init__(self, train_data, test_data, val_data):
        '''
        Initialize the model with the given train, test and validation data.

        @param train_data: train data, which is a tensorflow dataset
        @param test_data: test data, which is a tensorflow dataset
        @param val_data: validation data, which is a tensorflow dataset

        '''
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.val_dataset = val_data
        self.model = None

    def encode_words(self, vocab_size):
        '''
        Encodes the words into integers using TextVectorization from tensorflow.
        
        @param vocab_size: the size of the vocabulary
        @return: the encoder
        '''
        encoder = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size, ngrams = (1,2))
        encoder.adapt(self.train_dataset.map(lambda text, label: text))
        return encoder

    def create_model(self, encoder):
        '''
        Creates a keras RNN model. It has two stacked LSTM layers,
        followed by a dense layer.

        @param encoder: the encoder
        @return: None
        '''
        self.model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    def train_model(self, epochs):
        '''
        Trains the model with the training the data that was given in the constructor.

        The create method should be called before this method.

        @param epochs: the number of epochs
        @return: None
        '''
        def f1_metric(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
            return f1_val

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy', f1_metric])
    
        self.model.fit(self.train_dataset,
                        epochs=epochs,
                        validation_data=self.val_dataset,
                        validation_steps=30)

    def evaluate_model(self):
        '''
        Evaluates the model with the test data that was given in the constructor.
        Prints out the accuracy and f1 score.

        The fit method should be called before this method.

        @return: None
        '''
        test_loss, test_acc, test_f1 = self.model.evaluate(self.test_dataset)
        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))
        print('Test F1: {}'.format(test_f1))
    
    def predict_model(self):
        '''
        Returns the predictions of the model that was made on the test data.

        The fit method should be called before this method.

        @return: the predictions
        '''
        self.predictions = self.model.predict(self.test_dataset)
        return self.predictions
        
    def get_binary_predictions(self):
        '''
        Returns the binary predictions of the model.

        The predict method should be called before this method.

        @return: the binary predictions
        '''
        binary_predictions = []
        texts = []
        labels = []

        for z in list(self.test_dataset.as_numpy_iterator()):
            texts.append(z[0])
            labels.append(z[1])

        for i in self.predictions:
            i = i[0]
            if i < 0 :
                i = 0
            else:
                i = 1
            binary_predictions.append(i)
        
        return binary_predictions
