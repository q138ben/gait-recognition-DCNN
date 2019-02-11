
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from make_matrix import make_matrix_6by9
from phase_reorder_p5_CNN import phase_reorder
from barchart_accuracy_p5_CNN import bar_accuracy 
from plot_roc_CNN import plot_roc
from plot_confusion_matrix import plot_confusion_matrix

from DNM import CNN, LSTM_model



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
#
#
class_names = np.array(['LR','MS','TS','PSw','Sw'])
#
nb_classes = 5
img_rows, img_cols = 6, 9

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
#
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






model = CNN(X_train,Y_train)

score = model.evaluate(X_test, Y_test, verbose=0)

y_pred= model.predict_classes(X_test)

# convert class vectors to binary class matrices
y_preds_labels = np_utils.to_categorical(y_pred, nb_classes)

# print classification report
print(classification_report(Y_test, y_preds_labels))






# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.savefig('Confusion matrix, without normalization',dpi = 1200)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('Normalized confusion matrix',dpi = 1200)
plt.show()









(phase_1_test, phase_2_test, phase_3_test, phase_4_test,phase_5_test,phase_1_pred,phase_2_pred,phase_3_pred,phase_4_pred,phase_5_pred) = \
phase_reorder(y_test,y_pred)


print("\tOverall_Accuracy: %1.3f" % score[1])
print("\tLR_Accuracy: %1.3f" % accuracy_score(phase_1_test, phase_1_pred))
print("\tMS_Accuracy: %1.3f" % accuracy_score(phase_2_test, phase_2_pred))
print("\tTS_Accuracy: %1.3f" % accuracy_score(phase_3_test, phase_3_pred))
print("\tPSw_Accuracy: %1.3f" % accuracy_score(phase_4_test, phase_4_pred))
print("\tSw_Accuracy: %1.3f" % accuracy_score(phase_5_test, phase_5_pred))

CNN_acc = np.array([score[1],accuracy_score(phase_1_test, phase_1_pred),
                            accuracy_score(phase_2_test, phase_2_pred),accuracy_score(phase_3_test, phase_3_pred),
                            accuracy_score(phase_4_test, phase_4_pred),accuracy_score(phase_5_test, phase_5_pred)])


bar_accuracy(CNN_acc)
plot_roc(model,nb_classes,X_test,Y_test)