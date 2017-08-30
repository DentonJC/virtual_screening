#!/usr/bin/env python
"""
Logistic regression model definition
"""

from keras.models import Sequential
from keras.layers import Dense, merge, Input
#from keras.regularizers import WeightRegularizer

def build_logistic_model(input_dim, output_dim, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam', learning_rate=0.01, momentum=0, init_mode='uniform'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, kernel_initializer=init_mode, activation=activation, units=output_dim))
    optimizer = compile_optimizer(optimizer, learning_rate, momentum)
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
from src.main import compile_optimizer

def build_residual_model(input_dim, output_dim, activation_0='relu', activation_1='softmax', activation_2='sigmoid', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam', learning_rate=0.01, momentum=0, init_mode='uniform', dropout=0, layers=3):    
    input = Input(shape=(input_dim,))
    #embedded = Embedding(input_dim, input_dim)(input)

    def get_model():
        inputs = Input(shape=(input_dim,))
        x = Dense(input_dim, activation=activation_0)(inputs)
        x = Dense(input_dim, activation=activation_0)(x)
        predictions = Dense(input_dim, activation=activation_1)(x)
        return Model(inputs, predictions)
    
    resnet = residual_block(get_model())(input)
    for _ in range(layers):
        resnet = residual_block(get_model())(resnet)

    """maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False),
                     output_shape=lambda x: (x[0], x[2]))(resnet)"""
    dropout = Dropout(dropout)(resnet)
    
    
    output = Dense(output_dim, activation=activation_2)(dropout)
    model = Model(inputs=input, outputs=output)
    optimizer = compile_optimizer(optimizer, learning_rate, momentum)
    # try using different optimizers and different optimizer configs
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
    
    
def ann(l1=0.0, l2=0.0, dropout=0.0, dropout_in=0.0, hiddendim=4, hiddenlayers=3, lr=0.001, nb_epoch=50):
    model = Sequential()
    model.add(Dropout(dropout_in, input_shape=(X_train.shape[1],)))
    for i in range(hiddenlayers):
        #wr = WeightRegularizer(l2 = l2, l1 = l1) 
        model.add(Dense(output_dim=hiddendim, activation="relu"))#, W_regularizer=wr))
        model.add(Dropout(dropout))
    #wr = WeightRegularizer(l2 = l2, l1 = l1) 
    model.add(Dense(y_train.shape[1], activation='sigmoid'))#,W_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer = SGD(lr=lr, momentum=0.9, nesterov=True), metrics=['binary_crossentropy'])
    return model
