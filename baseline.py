from typing import Iterator, Iterable, Tuple, Text, Union


import random
import itertools
import pandas as pd
import csv
import numpy as np
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

NDArray = Union[np.ndarray, spmatrix]


class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        vectorizer = CountVectorizer(binary = True, ngram_range=(1,2))  # specify ngram range (1,2) to support multi-word expressions.
        vectorizer.fit(texts)
        self.features = vectorizer.vocabulary_ # use the vocabulary generated with CountVectorizer.
        

    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        return self.features[feature]


    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        vectorizer = CountVectorizer(vocabulary=self.features, binary=True, ngram_range=(1,2)) # Create a binary feature matrix and use the vocabulary of this object.
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray()



class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        """
        le = preprocessing.LabelEncoder()
        self.le = le.fit(labels)

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        """
        return list(self.le.transform([label]))[0]

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        return self.le.transform(labels)



class Classifier:
    def __init__(self):
        """Initalizes a logistic regression classifier.
        """
        self.clf = RandomForestClassifier() 


    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        print("Starting to train")
        self.clf.fit(features, labels)


    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        print("predicting")
        return self.clf.predict(features)
        

