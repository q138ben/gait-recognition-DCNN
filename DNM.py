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

from numpy import array
from keras.utils import np_utils
from sklearn.utils import class_weight
import numpy as np

def CNN(X_train,Y_train,X_test,Y_test,class_weights,subject):
    
    batch_size = 64
    nb_classes = 5
    nb_epoch = 100
    
    # input image dimensions
    img_rows, img_cols = 4,3
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
#    model.add(Conv2D(nb_filters, kernel_size))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=pool_size))
#    model.add(Dropout(0.25))

#    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(2*nb_filters, kernel_size,padding = 'same'))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=pool_size))
#    model.add(Dropout(0.25))




    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    
    opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    mc = ModelCheckpoint(subject+'_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    

    
    
    history = model.fit(X_train, Y_train,validation_data=(X_test, Y_test), batch_size=batch_size, nb_epoch=nb_epoch,verbose=0,callbacks=[es,mc],class_weight=None)
    model = load_model(subject+'_best_model.h5')
    
    return model, history


def MLP_(X_train,Y_train,X_test,Y_test,class_weights,subject):
    
    batch_size = 20
    nb_classes = 5
    nb_epoch = 100
    
    
    
    # define model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    #model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    mc = ModelCheckpoint(subject+'_best_MLP_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    

    
    history = model.fit(X_train, Y_train,validation_data=(X_test, Y_test), batch_size=batch_size, nb_epoch=nb_epoch,verbose=0,callbacks=[es,mc],class_weight=None)
    model = load_model(subject+'_best_MLP_model.h5')
    
    
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


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



def LSTM1(X_train,Y_train,n_steps,nb_classes,subject):
    # choose a number of time steps
    nb_epoch=30

    # the dataset knows the number of features, e.g. 2
    n_features = X_train.shape[2]
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
#    model.add(Dense(1)) # dense 1 for 'mse' loss
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
 #   model.compile(optimizer='adam', loss='mse') 
    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    mc = ModelCheckpoint(subject+'_best_LSTM_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    # fit model
#   model.fit(X_train, Y_train, epochs=20, verbose=1)
    model.fit(X_train, Y_train,validation_split=0.2, epochs=nb_epoch,verbose=0,callbacks=[es,mc])
    
    return model


def feature_selection(X):
    
    #feature_ind = (53,51,49,44,42,40,35,26,24,22,17,8)
    
    feature_ind = (53,51,49)
    
    feature_selec = np.zeros((np.shape(X)[0],len(feature_ind)))
    
    for i, e in enumerate(feature_ind):
        feature_selec[:,i]= X[:,e]
    
    
    return feature_selec
