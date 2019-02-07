
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import KFold

import tensorflow as tf
from make_matrix import make_matrix_6by9
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import numpy as np
from phase_reorder_p5_CNN import phase_reorder
from sklearn.metrics import accuracy_score


from barchart_accuracy_p5_CNN import bar_accuracy 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

#X_1 = genfromtxt('feature_5p_longbin_4.0kmh_X.txt', delimiter='');
#y_1 = genfromtxt('feature_5p_longbin_4.0kmh_Y.txt', delimiter='');
#
#X_2 = genfromtxt('feature_5p_longbin_4.5kmh_X.txt', delimiter='');
#y_2 = genfromtxt('feature_5p_longbin_4.5kmh_Y.txt', delimiter='');
#
#X_3 = genfromtxt('feature_5p_longbin_5.0kmh_X.txt', delimiter='');
#y_3 = genfromtxt('feature_5p_longbin_5.0kmh_Y.txt', delimiter='');
#
#
#X = np.concatenate((X_1,X_2,X_3))
#y = np.concatenate((y_1,y_2,y_3))



#X = genfromtxt('feature_5p_longbin_5.0kmh_normafter_excqua_X.txt', delimiter='');
#y = genfromtxt('feature_5p_longbin_5.0kmh_normafter_excqua_Y.txt', delimiter='');
#
#
#
#X = make_matrix_6by9(X)
#y = np.subtract(y,1)
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3,random_state=0)


#X = genfromtxt('feature_5p_longbin_5.0kmh_normafter_excqua_X.txt', delimiter='');
#y = genfromtxt('feature_5p_longbin_5.0kmh_normafter_excqua_Y.txt', delimiter='');
#X_train = make_matrix_6by9(X)
#y_train = np.subtract(y,1)
#X = genfromtxt('feature_5p_ben_normafter_excqua_X.txt', delimiter='');
#y = genfromtxt('feature_5p_ben_normafter_excqua_Y.txt', delimiter='');
#X_test = make_matrix_6by9(X)
#y_test = np.subtract(y,1)



X_1 = genfromtxt('feature_5p_longbin_4.0kmh_normafter_excqua_X.txt', delimiter='');
y_1 = genfromtxt('feature_5p_longbin_4.0kmh_normafter_excqua_Y.txt', delimiter='');

X_2 = genfromtxt('feature_5p_longbin_4.5kmh_normafter_excqua_X.txt', delimiter='');
y_2 = genfromtxt('feature_5p_longbin_4.5kmh_normafter_excqua_Y.txt', delimiter='');

X_3 = genfromtxt('feature_5p_longbin_5.0kmh_normafter_excqua_X.txt', delimiter='');
y_3 = genfromtxt('feature_5p_longbin_5.0kmh_normafter_excqua_Y.txt', delimiter='');

X_4 = genfromtxt('feature_5p_ben_normafter_excqua_X.txt', delimiter='');
y_4 = genfromtxt('feature_5p_ben_normafter_excqua_Y.txt', delimiter='');

X = np.concatenate((X_1,X_2,X_3,X_4))
y = np.concatenate((y_1,y_2,y_3,y_4))

X = make_matrix_6by9(X)
y = np.subtract(y,1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3,random_state=0)















batch_size = 200
nb_classes = 5
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 6, 9
# number of convolutional filters to use
nb_filters = 8
# size of pooling area for max pooling
pool_size = (2, 3)
# convolution kernel size
kernel_size = (2, 2)



if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)
model.evaluate(X_test, Y_test, verbose=0)


#function for converting predictions to labels
def prep_submissions(preds_array, file_name='abc.csv'):
    preds_df = pd.DataFrame(preds_array)
    predicted_labels = preds_df.idxmax(axis=1) #convert back one hot encoding to categorical variabless
    return predicted_labels
    '''
    ### prepare submissions in case you need to submit
    submission = pd.read_csv("test.csv")
    submission['label'] = predicted_labels
    submission.to_csv(file_name, index=False)
    print(pd.read_csv(file_name).head())
   '''
 #function to draw confusion matrix
def draw_confusion_matrix(true,preds):
    conf_matx = confusion_matrix(true, preds)
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
    plt.show()
    #return conf_matx  

y_pred= model.predict_classes(X_test)
#y_preds_labels = prep_submissions(y_pred)

y_preds_labels = np_utils.to_categorical(y_pred, nb_classes)
print(classification_report(Y_test, y_preds_labels))

draw_confusion_matrix(Y_test.argmax(axis=1), y_preds_labels.argmax(axis=1))












#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
#
#
#
#y_pred= model.predict_classes(X_test)
#
#
#(phase_1_test, phase_2_test, phase_3_test, phase_4_test,phase_5_test,phase_1_pred,phase_2_pred,phase_3_pred,phase_4_pred,phase_5_pred) = \
#phase_reorder(y_test,y_pred)
#
#print("\tLR_Accuracy: %1.3f" % accuracy_score(phase_1_test, phase_1_pred))
#print("\tMS_Accuracy: %1.3f" % accuracy_score(phase_2_test, phase_2_pred))
#print("\tTS_Accuracy: %1.3f" % accuracy_score(phase_3_test, phase_3_pred))
#print("\tPSw_Accuracy: %1.3f" % accuracy_score(phase_4_test, phase_4_pred))
#print("\tSw_Accuracy: %1.3f" % accuracy_score(phase_5_test, phase_5_pred))
#
#CNN_acc = np.array([score[1],accuracy_score(phase_1_test, phase_1_pred),
#                            accuracy_score(phase_2_test, phase_2_pred),accuracy_score(phase_3_test, phase_3_pred),
#                            accuracy_score(phase_4_test, phase_4_pred),accuracy_score(phase_5_test, phase_5_pred)])
#
#
#bar_accuracy(CNN_acc)