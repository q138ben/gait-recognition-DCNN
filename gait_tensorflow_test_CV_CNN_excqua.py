
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras import backend as K
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from make_matrix import make_matrix_6by9, make_matrix_7by9
from DNM import CNN, LSTM_model
from plot import phase_reorder,bar_accuracy,plot_roc,plot_confusion_matrix,plot_loss_and_acc
import time

## Subject snorri

#X_1 = genfromtxt('snorri_4.1_EmgRectifyLowPass6_IMULowPass6_X.txt', delimiter='');
#y_1 = genfromtxt('snorri_4.1_EmgRectifyLowPass6_IMULowPass6_Y.txt', delimiter='');
#
#X_2 = genfromtxt('snorri_4.8_EmgRectifyLowPass6_IMULowPass6_X.txt', delimiter='');
#y_2 = genfromtxt('snorri_4.8_EmgRectifyLowPass6_IMULowPass6_Y.txt', delimiter='');
#
#X_3 = genfromtxt('snorri_5.4_EmgRectifyLowPass6_IMULowPass6_X.txt', delimiter='');
#y_3 = genfromtxt('snorri_5.4_EmgRectifyLowPass6_IMULowPass6_Y.txt', delimiter='');
#
#X_4 = genfromtxt('snorri_6.0_EmgRectifyLowPass6_IMULowPass6_X.txt', delimiter='');
#y_4 = genfromtxt('snorri_6.0_EmgRectifyLowPass6_IMULowPass6_Y.txt', delimiter='');
#
#X_5 = genfromtxt('snorri_6.6_EmgRectifyLowPass6_IMULowPass6_X.txt', delimiter='');
#y_5 = genfromtxt('snorri_6.6_EmgRectifyLowPass6_IMULowPass6_Y.txt', delimiter='');
#


## subject macus

#X_1 = genfromtxt('Marcus_6.4_X.txt', delimiter='');
#y_1 = genfromtxt('Marcus_6.4_Y.txt', delimiter='');
#
#X_2 = genfromtxt('Marcus_5.8_X.txt', delimiter='');
#y_2 = genfromtxt('Marcus_5.8_Y.txt', delimiter='');
#
#X_3 = genfromtxt('Marcus_5.2_X.txt', delimiter='');
#y_3 = genfromtxt('Marcus_5.2_Y.txt', delimiter='');
#
#X_4 = genfromtxt('Marcus_4.6_X.txt', delimiter='');
#y_4 = genfromtxt('Marcus_4.6_Y.txt', delimiter='');
#
#X_5 = genfromtxt('Marcus_4.0_X.txt', delimiter='');
#y_5 = genfromtxt('Marcus_4.0_Y.txt', delimiter='');
#
#X = np.concatenate((X_1,X_2,X_3,X_4,X_5))
#y = np.concatenate((y_1,y_2,y_3,y_4,y_5))




#X = genfromtxt('Marcus_6.4_X.txt', delimiter='');
#y = genfromtxt('Marcus_6.4_Y.txt', delimiter='');



#X_train = np.concatenate((X_1,X_2,X_3,X_4))
#y_train = np.concatenate((y_1,y_2,y_3,y_4))
#
#X_test = X_5
#y_test = y_5
#
#X_train = X_train[:,8:62]
#X_test = X_test[:,8:62]
#X_train = make_matrix_6by9(X_train)
#X_test = make_matrix_6by9(X_test)
#y_train = np.subtract(y_train,1)
#y_test = np.subtract(y_test,1)


#X = np.concatenate((X[:,:18],X[:, 27:]),axis = 1)
#X = X[:,8:]  # selecting IMU 9-14


X = genfromtxt('yixing_5.5_X.txt', delimiter='');
y = genfromtxt('yixing_5.5_Y.txt', delimiter='');

X = X[:,8:];

X = make_matrix_7by9(X)
y = np.subtract(y,1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3,random_state=0)


#X_train = X[:7000,:,:]
#X_test = X[7000:,:,:]
#y_train = y[:7000]
#y_test = y[7000:]




#
#
class_names = np.array(['LR','MS','TS','PSw','Sw'])
#
nb_classes = 5
img_rows, img_cols = 7, 9

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
#X_train /= 255
#X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)






model,history  = CNN(X_train,Y_train,X_test,Y_test)

score = model.evaluate(X_test, Y_test, verbose=0)


# ----------Caculate the computational time for test data

start = time.clock() 
y_pred= model.predict_classes(X_test)
end = time.clock()
ct = end - start
ctf = ct/len(y_test)
print ('computational time for each frame = ',ctf)

# convert class vectors to binary class matrices
y_preds_labels = np_utils.to_categorical(y_pred, nb_classes)

# print classification report
print(classification_report(Y_test, y_preds_labels))






# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)

cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
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




#plot bar_accuracy

CNN_acc = np.array([score[1],cnf_matrix_norm[0,0],cnf_matrix_norm[1,1],cnf_matrix_norm[2,2],cnf_matrix_norm[3,3],cnf_matrix_norm[4,4]])
bar_accuracy(CNN_acc)

# plot roc curve

plot_roc(model,nb_classes,X_test,Y_test)

#plot loss and accuracy on training and validation data

plot_loss_and_acc(history)