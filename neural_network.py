import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K
from time import time


class FeedForwardNetwork:
    def __init__(self, hidden_layers=[50], dropout=0, activation='softmax', optimizer='adam', metrics=['accuracy'],
                 loss='categorical_crossentropy', epochs=100, batch_size=128, timeit=True, verbosity=1, callbacks=[], class_weight=None, 
                 validation_split=0.0, validation_data=None):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.activation = activation
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.timeit = timeit
        self.verbosity = verbosity
        self.callbacks = callbacks
        self.class_weight = class_weight
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.model = Sequential()

    def fit(self, X, y):
        if self.timeit:
            start_time = time()
        n_features = X.shape[1]
        if self.verbosity:
            print('Data size ({:d}, {:d}) -\t Epochs {:d} -\t Batch Size {:d}'.format(X.shape[0], X.shape[1], self.epochs, self.batch_size))
        try:
            n_classes = y.shape[1]
            if self.class_weight=='balanced':
            	weights = list(y.shape[0] / (n_classes * y.sum(axis=0)))
            	self.class_weight = {i:weights[i] for i in range(len(weights))}
            	print('Computed Class Weights', self.class_weight)
        except IndexError:
            n_classes = 1
            self.loss = 'binary_crossentropy'
            self.activation = 'sigmoid'
            if self.class_weight=='balanced':
            	weights = list(y.shape[0] / (2 * np.bincount(y)))
            	self.class_weight = {0: weights[0], 1: weights[1]}
            	print('Computed Class Weights', self.class_weight)
        K.clear_session()
        self.model.add(Dense(units=self.hidden_layers[0], input_shape=(n_features,), name='Hidden_0'))
        for i, h in enumerate(self.hidden_layers[1:]):
            self.model.add(Dense(units=h, activation='relu', name=f'Hidden_{i+1}'))
        if self.dropout:
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(units=n_classes, activation=self.activation, name=f'Output_{self.activation}'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.fit(X, y, verbose=(self.verbosity-1), epochs=self.epochs, batch_size=self.batch_size, 
        				callbacks=self.callbacks, class_weight=self.class_weight, validation_data=self.validation_data, 
        				validation_split=self.validation_split)
        if self.timeit:
            print('Fit complete in {:.2f} seconds'.format(time()-start_time))

    def predict(self, X):
        return np.array(self.model.predict(X, batch_size=self.batch_size) > 0.5, dtype=np.uint8)

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=self.batch_size)
