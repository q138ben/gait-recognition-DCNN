# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:27:48 2019

@author: binbi
"""

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import LSTM

from keras.layers import Conv2D

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def CNN(X_train,Y_train,X_test,Y_test):
    
    batch_size = 32
    nb_classes = 5
    nb_epoch = 50
    
    # input image dimensions
    img_rows, img_cols = 7,9
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling,default (2,2)
    pool_size = (2, 2)
    # convolution kernel size, default (3,3)
    kernel_size = (3, 3)
    
    
    
    model = Sequential()
    

#
    model.add(Conv2D(nb_filters, kernel_size,input_shape=(img_rows, img_cols, 1),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))




    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    
    opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 5)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    
    
    history = model.fit(X_train, Y_train,validation_data=(X_test, Y_test), batch_size=batch_size, nb_epoch=nb_epoch,verbose=0,callbacks=[es,mc])
    model = load_model('best_model.h5')
    
    return model, history


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

