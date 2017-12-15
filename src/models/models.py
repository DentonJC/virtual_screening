#!/usr/bin/env python

import os
import sys
# import keras.backend as K  # for max pooling operation
# from keras import regularizers  # regularizers.l1_l2(0.)
# from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input  # , Embedding, LSTM, merge, Bidirectional, Lambda
from keras.layers import Embedding, LSTM, Merge, TimeDistributed, merge, GRU, SimpleRNN
from src.models.residual_blocks import residual_block
from src.main import compile_optimizer


def build_logistic_model(input_dim, output_dim, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam', learning_rate=0.01, momentum=0, init_mode='uniform'):
    """
    Logistic regression model definition
    """
    model = Sequential()
    model.add(Dense(input_dim=input_dim, kernel_initializer=init_mode, activation=activation, units=output_dim))
    optimizer = compile_optimizer(optimizer, learning_rate, momentum)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def build_residual_model(input_dim, output_dim, activation_0='relu', activation_1='softmax', activation_2='sigmoid', loss='binary_crossentropy',
                         metrics=['accuracy'], optimizer='Adam', learning_rate=0.01, momentum=0, init_mode='uniform', dropout=0, layers=3):
    """
    Fully connected residual model definition
    """
    inp = Input(shape=(input_dim,))
    # embedded = Embedding(input_dim, input_dim)(input)

    def get_model():
        inputs = Input(shape=(input_dim,))
        x = Dense(input_dim, activation=activation_0, kernel_initializer=init_mode)(inputs)
        x = Dense(input_dim, activation=activation_0, kernel_initializer=init_mode)(x)
        predictions = Dense(input_dim, activation=activation_1, kernel_initializer=init_mode)(x)
        return Model(inputs, predictions)

    resnet = residual_block(get_model())(inp)
    for _ in range(layers):
        resnet = residual_block(get_model())(resnet)

    # maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))(resnet)
    dropout = Dropout(dropout)(resnet)

    output = Dense(output_dim, activation=activation_2, kernel_initializer=init_mode)(dropout)
    model = Model(inputs=inp, outputs=output)
    optimizer = compile_optimizer(optimizer, learning_rate, momentum)
    # try using different optimizers and different optimizer configs
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def lstm(input_dim = 28, MAX_SEQ_LENGTH=None, N_CLASSES=2):
    # forward
    encoder_a = Sequential()
    encoder_a.add(LSTM(8, input_dim=input_dim,return_sequences=True))
    # backward
    encoder_b = Sequential()
    encoder_b.add(LSTM(8, input_dim=input_dim,go_backwards=True,return_sequences=True))

    model = Sequential()
    model.add(Merge([encoder_a, encoder_b], mode='concat'))
    model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    return model


if __name__ == "__main__":
    """
    Print models summary in file.
    """
    path = os.path.dirname(os.path.realpath(__file__)).replace("src/models", "") + "/tmp/"
    input_dim = 256
    output_dim = 2
    
    print("Residual")
    f = open(path + 'model', 'w')
    # print and save model summary
    model = build_residual_model(input_dim, output_dim)
    orig_stdout = sys.stdout
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    
    print("Regression")
    model = build_logistic_model(input_dim, output_dim)
    orig_stdout = sys.stdout
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout    
    f.close()
