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
from fetchDataset import fetchDataset
from keras.models import load_model

import os
#os.chdir("C:\\Users\\binbi\\Desktop\\data_aquisition\\yazu_05_08")
#os.chdir('C:\\Users\\binbi\\Desktop\\data_aquisition\\yixing')
#os.chdir("D:\\data_aquisition\\snorri")
#os.chdir("D:\\data_aquisition\\gunnar")

#
#X1,y1= fetchDataset('gunnar')
#X2,y2= fetchDataset('hui')
#X3,y3= fetchDataset('Marcus')
#X4,y4= fetchDataset('yanzu')
#X5,y5= fetchDataset('yixing')
#X6,y6= fetchDataset('snorri')
#
#
#os.chdir("D:\\data_aquisition\\snorri")
#
#
#
#X_train = np.concatenate((X1,X2,X3,X4,X5))
#y_train = np.concatenate((y1,y2,y3,y4,y5))
##
#X_test = X6
#y_test = y6
#
#X_train = X_train[:,8:]
#X_test = X_test[:,8:]
#X_train = make_matrix_7by9(X_train)
#X_test = make_matrix_7by9(X_test)
#y_train = np.subtract(y_train,1)
#y_test = np.subtract(y_test,1)



#
#X_train = X_train[:,8:62]
#X_test = X_test[:,8:62]
#X_train = make_matrix_6by9(X_train)
#X_test = make_matrix_6by9(X_test)
#y_train = np.subtract(y_train,1)
#y_test = np.subtract(y_test,1)


#X = np.concatenate((X[:,:18],X[:, 27:]),axis = 1)
#X = X[:,8:]  # selecting IMU 9-14





X,y= fetchDataset('gunnar_4.2')

X = X[:,8:];

X = make_matrix_7by9(X)
y = np.subtract(y,1)




X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3,random_state=0)
#

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






#model,history  = CNN(X_train,Y_train,X_test,Y_test)
model = load_model('best_model.h5')

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








#plt.savefig('Confusion matrix, without normalization',dpi = 1200)

# Plot normalized confusion matrix

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Subject 1 (4.2km/h)')
plt.savefig('gunnar_4.2_confusion_matrix.png',dpi = 400)



plt.show()




#plot bar_accuracy

CNN_acc = np.array([score[1],cnf_matrix_norm[0,0],cnf_matrix_norm[1,1],cnf_matrix_norm[2,2],cnf_matrix_norm[3,3],cnf_matrix_norm[4,4]])

#CNN_onevsRest=np.concatenate((CNN_onevsRest,CNN_acc),axis=0)


#bar_accuracy(CNN_acc)


'''

# plot roc curve

#plot_roc(model,nb_classes,X_test,Y_test)

#plot loss and accuracy on training and validation data

plot_loss_and_acc(history)
'''