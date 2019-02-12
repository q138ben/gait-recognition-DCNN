# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:27:48 2019

@author: binbi
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import LSTM

from keras.layers import Conv2D


def CNN(X_train,Y_train):
    
    batch_size = 200
    nb_classes = 5
    nb_epoch = 30
    
    # input image dimensions
    img_rows, img_cols = 6, 9
    # number of convolutional filters to use
    nb_filters = 16
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (2, 2)
    
    
    model = Sequential()
    


    model.add(Conv2D(nb_filters, kernel_size,input_shape=(img_rows, img_cols, 1),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

#    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=pool_size))
#    model.add(Dropout(0.25))





    model.add(Flatten())
#    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)
    return model


def LSTM_model(X_train,Y_train):
    
    max_features = 1024
    nb_classes = 5
    batch_size=200
    epochs=1

    
    
    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
    return model

