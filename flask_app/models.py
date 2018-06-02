"""
    File name: models.py
    Author: Kshitij Shah
    Date created: 4/21/2018
    Python Version: 3.6
"""

from keras.models import load_model
import pickle as pk
import numpy as np


class FFNPredictor:

    def __init__(self, model_path, vectorizer_path):
        """
        Loads the pretrained feed forward network model and vectorizer.
        Provides the predict method to predict label when a query is made.
        :param model_path: path to the feed forward network model
        :param vectorizer_path: path to vectorizer
        """
        self.model = load_model(model_path)
        self.vectorizer = pk.load(open(vectorizer_path, 'rb'))
        self.labels = [x.title() for x in
                       ['APPLICATION',
                        'BILL',
                        'BILL BINDER',
                        'BINDER',
                        'CANCELLATION NOTICE',
                        'CHANGE ENDORSEMENT',
                        'DECLARATION',
                        'DELETION OF INTEREST',
                        'EXPIRATION NOTICE',
                        'INTENT TO CANCEL NOTICE',
                        'NON-RENEWAL NOTICE',
                        'POLICY CHANGE',
                        'REINSTATEMENT NOTICE',
                        'RETURNED CHECK']
                       ]

    def predict(self, words):
        """
        Predict the label and confidence given the words from a document.
        :param words: Words as a space separated string
        :return: predicted label and confidence in the prediction
        """
        pred = self.model.predict(self.vectorizer.transform([words]))
        return self.labels[np.argmax(pred)], np.max(pred)
