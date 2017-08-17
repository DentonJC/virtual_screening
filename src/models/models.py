#!/usr/bin/env python
"""
Logistic regression model definition
"""

from keras.models import Sequential
from keras.layers import Dense, merge, Input

def build_logistic_model(input_dim, output_dim, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, activation=activation, units=output_dim))
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model

"""
Fully connected residual model definition
"""
import numpy as np

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional, Lambda
from keras.datasets import imdb
import keras.backend as K  # Needed for max pooling operation
from src.models.residual_blocks import residual_block

def build_residual_model(input_dim, output_dim, ativation_0='relu', activation_1='softmax', activation_2='sigmoid', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):    
    input = Input(shape=(input_dim,))
    embedded = Embedding(input_dim, input_dim)(input)

    def get_model():
        inputs = Input(shape=(input_dim,))
        x = Dense(input_dim, activation_0)(inputs)
        x = Dense(input_dim, activation_0)(x)
        predictions = Dense(input_dim, activation=activation_1)(x)
        return Model(inputs, predictions)

    resnet = residual_block(get_model())(embedded)
    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False),
                     output_shape=lambda x: (x[0], x[2]))(resnet)
    dropout = Dropout(0.5)(maxpool)
    output = Dense(output_dim, activation=activation_2)(dropout)
    model = Model(inputs=input, outputs=output)

    # try using different optimizers and different optimizer configs
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
