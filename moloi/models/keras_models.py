#!/usr/bin/env python

import os
import sys
import keras
from keras.optimizers import Adam, Nadam, Adamax, RMSprop, Adagrad, Adadelta, SGD
from keras.engine import InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Embedding, Merge, TimeDistributed, merge, SimpleRNN, RepeatVector, Wrapper, BatchNormalization, Activation
from keras.layers import GRU as GRU_layer
from keras.layers import LSTM as LSTM_layer
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, RemoteMonitor, ReduceLROnPlateau
from keras.regularizers import l2 as l2_reg


def create_callbacks(output, patience, section, monitor='val_acc', mode='auto', callbacks="[stopping, csv_logger, checkpoint, remote, lr]", factor=0.1):           
    filepath = output + "results/weights-improvement.hdf5"
    path = output + 'history_' + os.path.basename(sys.argv[0]).replace(".py", "") + "_" + str(section) + '.csv'
    
    checkpoint = ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=True, mode=mode)
    stopping = EarlyStopping(monitor=monitor, min_delta=0.00001, patience=patience, verbose=1, mode=mode)
    csv_logger = CSVLogger(path, append=True, separator=';')
    remote = RemoteMonitor(root='http://localhost:8080', path=output, field='data', headers=None)
    lr = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=int(patience/3), verbose=1, mode=mode, cooldown=0, min_lr=0)
    callbacks_list = eval(callbacks)
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


def Logreg(input_shape, output_shape, l2=0.0, lr=0.1, momentum=0.9, metrics=['binary_crossentropy', 'accuracy'], loss='binary_crossentropy'):
    """
    Logistic regression model definition
    """
    model = Sequential()
    model.add(Dense(input_shape=(input_shape, ), activation="sigmoid", kernel_regularizer=l2_reg(l2), units=output_shape))
    model.compile(loss=loss, optimizer=SGD(lr=lr, momentum=momentum), metrics=metrics)
    return model


def MultilayerPerceptron(input_shape, output_shape, activation_1='sigmoid', activation_2='sigmoid', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam', learning_rate=0.01, momentum=0, init_mode='uniform', layers=3, neurons_1=10, neurons_2=10):
    """
    Multilayer perceptron model definition
    """
    model = Sequential()
    model.add(Dense(input_shape=(input_shape, ), kernel_initializer=init_mode, activation=activation_1, units=neurons_1))

    for _ in range(layers):
        model.add(Dense(kernel_initializer=init_mode, activation=activation_1, units=neurons_2))

    model.add(Dense(kernel_initializer=init_mode, activation=activation_2, units=output_shape))
    optimizer = compile_optimizer(optimizer, learning_rate, momentum)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def FCNN(input_shape, output_shape, inference=False, input_dropout=0.0, l2=0.0, hidden_dim_1=100, hidden_dim_2=100, activation="relu", bn=True, dropout_1=0.3, dropout_2=0.3, loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    dropouts = [dropout_1, dropout_2]
    hidden_dims = [hidden_dim_1, hidden_dim_2]
    input = Input(shape=(input_shape,))
    x = Dropout(input_dropout)(input, training=not inference)
    for h_id, (hd, drop) in enumerate(zip(hidden_dims, dropouts)):
        x = Dense(hd, name="dense_" + str(h_id), kernel_regularizer=l2_reg(l2))(x)
        if bn:
            x = BatchNormalization(name="bn_" + str(h_id))(x)
        x = Dropout(drop, name="drop_" + str(h_id))(x) #, training=not inference)
        x = Activation(activation, name="act_" + str(h_id))(x)
    # TODO: Refactor
    if output_shape == 1:
        output = Dense(input_shape=(input_shape,), activation="sigmoid", units=1, name="final_softmax",
            kernel_regularizer=l2_reg(l2))(x)
    else:
        output = Dense(input_shape=(input_shape,), activation="softmax", units=output_shape, name="final_softmax",
            kernel_regularizer=l2_reg(l2))(x)
    model = Model(inputs=[input], outputs=[output])
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    #model.summary()
    setattr(model, "steerable_variables", {})
    return model


def LSTM(input_shape, output_shape, input_length, embedding_length=64, neurons=256, activation='softmax', loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam'):
    model = Sequential()
    model.add(Embedding(input_shape, embedding_length, input_length=input_length))
    model.add(LSTM_layer(neurons))
    model.add(Dropout(0.2))    
    model.add(Dense(output_shape, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


if __name__ == "__main__":
    """
    Print models summary.
    """
    input_shape = 256
    output_shape = 2
    input_length = 128
    
    print("Logreg")
    model = Logreg(input_shape, output_shape)
    print(model.summary())
    
    print("MultilayerPerceptron")
    model = MultilayerPerceptron(input_shape, output_shape)
    print(model.summary())
    
    print("FCNN")
    model = FCNN(input_shape, output_shape)
    print(model.summary())

    print("LSTM")
    model = LSTM(input_shape, output_shape, input_length)  
    print(model.summary())
