# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:47:45 2020

@author: binbi
"""
from fetchDataset import fetchDataset
from make_matrix import make_matrix_6by9, make_matrix_7by9,make_matrix_6by3
import numpy as np
import random
import DNM
from DNM import feature_selection
from keras.utils import np_utils
from sklearn.utils import class_weight
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from plot import plot_confusion_matrix
from sklearn.svm import LinearSVC


class gait_phase_recognition():
    def __init__(self, subject, classifier, is_CNN_classifer = True, is_MLP_classifier = False,img_rows =7, img_cols = 9,feature_ind = False):
        self.subject = subject
        self.classifier = classifier
        self.is_CNN_classifer= is_CNN_classifer
        self.is_MLP_classifer = is_MLP_classifier
        self.feature_ind = feature_ind
        self.img_rows = img_rows
        self.img_cols = img_cols
        
    def split_train_and_test(self, split = 0.7,nb_classes = 5, random_seed = 99):
        X,y= fetchDataset(self.subject)
        #X= X[:,8:] # extract raw data if the dataset includes quaternion
        if self.feature_ind:
            X = feature_selection(X,self.feature_ind)
        if self.is_CNN_classifer:
            [m,n] = np.shape(X)
            X_m_c = np.zeros([m,self.img_rows,self.img_cols])
            for i in range(m):
                X_m = X[i,:].reshape(self.img_rows,self.img_cols)
                X_m_c[i,:,:] = X_m
            X = X_m_c
        y = np.subtract(y,1)
        splitSize = int(split*len(y))
        random.seed(random_seed) #seed 8,9
        splitIndStart = random.randint(0,splitSize)
        splitIndEnd = splitIndStart+ len(y)-splitSize
        
        X_train = np.concatenate((X[:splitIndStart,:],X[splitIndEnd:,:]),axis=0)
        X_test = X[splitIndStart:splitIndEnd,:]
        Y_train = np.concatenate((y[:splitIndStart],y[splitIndEnd:]),axis = 0)
        Y_test = y[splitIndStart:splitIndEnd]
        
        
        
        if self.is_CNN_classifer:
            X_train = X_train.reshape(X_train.shape[0], self.img_rows, self.img_cols, 1).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], self.img_rows, self.img_cols, 1).astype('float32')
            Y_train = np_utils.to_categorical(Y_train, nb_classes)
            Y_test = np_utils.to_categorical(Y_test, nb_classes)
        if self.is_MLP_classifer:
            Y_train = np_utils.to_categorical(Y_train, nb_classes)
            Y_test = np_utils.to_categorical(Y_test, nb_classes)
        
        
        
        return X_train, Y_train, X_test,Y_test
    
    
    def train(self):
        X_train, Y_train, X_test,Y_test = self.split_train_and_test()
        
        if self.is_CNN_classifer:
        
            model, _  = self.classifier(X_train,Y_train,X_test,Y_test,img_rows=self.img_rows, img_cols=self.img_cols, class_weights=None, subject= self.subject)
            return model
        elif self.is_MLP_classifer:
            
            model,_ = self.classifier(X_train,Y_train,X_test,Y_test, class_weights=None, subject= self.subject)
            
            return model
            
        else:
            

            self.classifier.fit(X_train, Y_train)
        
        
    
    def test(self, model, load_mode = False, model_name= None, verbose= 0, save_text = False, transition_correction = True):
        _,_ ,X_test, y_test = self.split_train_and_test()
        if self.is_CNN_classifer or self.is_MLP_classifer:
            
            y_test = y_test.argmax(1) # reverse one hot encoding to label vector
            
        
            if load_mode:
                assert model_name, 'model name should not be empty'
                from keras.models import load_model
                model = load_model(model_name)
            
            y_pred= model.predict_classes(X_test) 
            
            
        else:
            y_pred = self.classifier.predict(X_test)
        
        if transition_correction:
            
            err_ind = [i for i, e in enumerate(y_test) if e != y_pred[i]] # get the index of falsely predicted gait phases
            y_diff = y_test - y_pred
            for _ ,e in enumerate(err_ind):
                if y_diff[e-1] == y_diff[e+1] == 0 and abs(y_diff[e]) == 1:
                    y_pred[e] = y_test[e]
                    
                    
        #y_preds_labels = np_utils.to_categorical(y_pred, 5)
        #print(classification_report(Y_test, y_preds_labels))
        acc = accuracy_score(y_test, y_pred)
        
        if save_text:
            np.savetxt(self.subject+'_score.csv',acc,delimiter='')
        
        return acc,y_pred,y_test

if __name__== "__main__":
    #feature_ind = (53,51,49,44,42,40,35,33,31,26,24,22,17,15,13,8,6,4)
    feature_ind = [18+i for i in range(9)]
    feature_ind.extend([45+i for i in range(9)])
    img_rows, img_cols = 3,6
    
    subjects = ('qing_frontal_3.7','qing_frontal_4.2','qing_frontal_4.7','qing_frontal_5.2','qing_frontal_5.7')
    CNN_acc = np.zeros((len(subjects),6))
    for i, subject in enumerate(subjects):
        
        gait = gait_phase_recognition(subject, DNM.CNN, is_CNN_classifer = True,img_rows =img_rows, img_cols = img_cols, feature_ind = feature_ind)
        #gait = gait_phase_recognition('gunnar_3.6', DNM.MLP_, is_CNN_classifer = False, is_MLP_classifier = True, feature_ind = feature_ind)
        X_train, Y_train, X_test,Y_test = gait.split_train_and_test()
        model = gait.train()
        
        acc,y_pred,y_test = gait.test(model, load_mode = False, model_name ='MLP_1FC512_longbin_3.8_best_MLP_model.h5', transition_correction=False)
        #acc,y_pred,y_test = gait.test(model, load_mode = False, model_name ='MLP_1FC512_longbin_3.8_best_MLP_model.h5', transition_correction=True)
        
        cnf_matrix = confusion_matrix(y_test, y_pred)
        cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)
        
        acc_speed = np.array([acc,cnf_matrix_norm[0,0],cnf_matrix_norm[1,1],cnf_matrix_norm[2,2],cnf_matrix_norm[3,3],cnf_matrix_norm[4,4]])
        CNN_acc[i:]=acc_speed
        
    np.savetxt('qing'+'_intra_1_acc.csv', CNN_acc, delimiter=",")
    #np.savetxt('qing_transition_correction'+'_intra_1_acc.csv', CNN_acc, delimiter="")
    
#    plot_confusion_matrix(cnf_matrix, classes=np.array(['LR','MS','TS','PSw','Sw']),
#                      title='Confusion matrix, without normalization,no feature selection')
    
    