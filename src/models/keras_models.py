#!/usr/bin/env python

import os
import sys
import keras
from keras.optimizers import Adam, Nadam, Adamax, RMSprop, Adagrad, Adadelta, SGD
from keras.engine import InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Embedding, Merge, TimeDistributed, merge, SimpleRNN, RepeatVector, Wrapper
from keras.layers import GRU as GRU_layer
from keras.layers import LSTM as LSTM_layer
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger


def create_callbacks(output, patience, section):
    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(output+"results/*"):
        os.makedirs(output+"results/")
            
    filepath = output + "results/weights-improvement.hdf5"
    ## error when just acc
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='min')
    stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=0, mode='auto')

    csv_logger = CSVLogger(output + 'history_' + os.path.basename(sys.argv[0]).replace(".py", "") +
                           "_" + str(section) + '.csv', append=True, separator=';')
    callbacks_list = [stopping, csv_logger] #checkpoint
    return callbacks_list


def compile_optimizer(optimizer, learning_rate=0.1, momentum=0.1):
    if optimizer == 'Adam':
        return Adam(lr=learning_rate)
    elif optimizer == 'Nadam':
        return Nadam(lr=learning_rate)
    elif optimizer == 'Adamax':
        return Adamax(lr=learning_rate)
    elif optimizer == 'RMSprop':
        return RMSprop(lr=learning_rate)
    elif optimizer == 'Adagrad':
        return Adagrad(lr=learning_rate)
    elif optimizer == 'Adadelta':
        return Adadelta(lr=learning_rate)
    else:
        return SGD(lr=learning_rate, momentum=momentum)


class residual_block(Wrapper):
    def build(self, input_shape):
        output_shape = input_shape
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = [InputSpec(shape=input_shape)]
        super(residual_block, self).build()

    def call(self, x, mask=None):
        layer_output = self.layer.call(x, mask)
        output = keras.layers.Add()([x, layer_output])
        return output


def Perceptron(input_shape, output_shape, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam', learning_rate=0.01, momentum=0, init_mode='uniform'):
    """
    Logistic regression model definition
    """
    model = Sequential()
    model.add(Dense(input_shape=(input_shape, ), kernel_initializer=init_mode, activation=activation, units=output_shape))
    optimizer = compile_optimizer(optimizer, learning_rate, momentum)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def Residual(input_shape, output_shape, activation_0='relu', activation_1='softmax', activation_2='sigmoid', loss='binary_crossentropy',
                         metrics=['accuracy'], optimizer='Adam', learning_rate=0.01, momentum=0, init_mode='uniform', dropout=0, layers=3):
    """
    Fully connected residual model definition
    """
    inp = Input(shape=(input_shape, ))
    #embedded = Embedding(input_shape, input_shape)(inp)

    def get_model():
        inputs = Input(shape=(input_shape, ))
        x = Dense(input_shape, activation=activation_0, kernel_initializer=init_mode)(inputs)
        x = Dense(input_shape, activation=activation_0, kernel_initializer=init_mode)(x)
        predictions = Dense(units=input_shape, activation=activation_1, kernel_initializer=init_mode)(x)
        return Model(inputs, predictions)

    resnet = residual_block(get_model())(inp)
    #resnet = residual_block(get_model())(embedded)
    for _ in range(layers):
        resnet = residual_block(get_model())(resnet)

    # maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))(resnet)
    dropout = Dropout(dropout)(resnet)

    output = Dense(output_shape, activation=activation_2, kernel_initializer=init_mode)(dropout)
    mod = Model(inputs=inp, outputs=output)
    
    model = Sequential()
    model.add(mod)
    
    optimizer = compile_optimizer(optimizer, learning_rate, momentum)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def LSTM_OLD(input_shape, output_shape, input_length, embedding_length=64, neurons=256, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()
    model.add(Embedding(input_shape, embedding_length, input_length=input_length))
    model.add(LSTM_layer(neurons))
    model.add(Dropout(0.2))    
    model.add(Dense(output_shape, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
    

#X shape=(number_of_documents, n_rows, n_cols)
#Y shape=(number_of_documents, num_categories)
def LSTM(input_shape, output_shape, input_length, embedding_length=64, neurons=256, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()

    model.add(LSTM(int(embedding_length), input_shape=input_shape)) #(n_rows, n_cols)))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape))
    model.add(Activation(activation))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)




def MLSTM(input_shape, output_shape, batch_size, layers=3, neurons_1=256, neurons_2=512, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()
    model.add(LSTM(neurons_1, return_sequences=True, stateful=True, 
               batch_input_shape=(batch_size, 1, input_shape)))
    for _ in range(layers):
        model.add(LSTM(neurons_2, return_sequences=True, stateful=True))
    model.add(LSTM(neurons_1, stateful=True))

    model.add(Dense(output_shape, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
    
    
def RNN(input_shape, output_shape, input_length, embedding_length=64, neurons=256, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()
    model.add(Embedding(input_shape, embedding_length, input_length=input_length))
    model.add(SimpleRNN(neurons))
    
    model.add(Dense(output_shape, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def MRNN(input_shape, output_shape, batch_size, layers=3, neurons_1=256, neurons_2=512, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()
    model.add(SimpleRNN(neurons_1, return_sequences=True, stateful=True, 
               batch_input_shape=(batch_size, 1, input_shape)))
    for _ in range(layers):
        model.add(SimpleRNN(neurons_2, return_sequences=True, stateful=True))
    model.add(SimpleRNN(neurons_1, stateful=True))

    model.add(Dense(output_shape, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def GRU(input_shape, output_shape, input_length, embedding_length=64, neurons=256, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()
    model.add(Embedding(input_shape, embedding_length, input_length=input_length))
    model.add(GRU_layer(neurons))
    
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


if __name__ == "__main__":
    """
    Print models summary.
    """
    input_shape = 256
    output_shape = 2
    input_length = 128
    
    print("Residual")
    model = Residual(input_shape, output_shape)
    print(model.summary())
   
    print("Perceptron")
    model = Perceptron(input_shape, output_shape)
    print(model.summary())
    
    print("RNN")
    model = RNN(input_shape, output_shape, input_length) 
    print(model.summary())
        
    print("GRU")
    model = GRU(input_shape, output_shape, input_length)  
    print(model.summary())
        
    print("LSTM")
    model = LSTM(input_shape, output_shape, input_length)  
    print(model.summary())

