# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:35:55 2018

@author: binbi
"""
import numpy as np

def make_matrix_6by13(X):
    
    [m,n] = np.shape(X)
    X_m_c = np.zeros([m,6,13])
    for i in range(m):
        X_m = X[i,:].reshape(6,13)
        X_m_c[i,:,:] = X_m
    
    return X_m_c
        
def make_matrix_6by9(X):
    
    [m,n] = np.shape(X)
    X_m_c = np.zeros([m,6,9])
    for i in range(m):
        X_m = X[i,:].reshape(6,9)
        X_m_c[i,:,:] = X_m
    
    return X_m_c

def make_matrix_7by9(X):
    
    [m,n] = np.shape(X)
    X_m_c = np.zeros([m,7,9])
    for i in range(m):
        X_m = X[i,:].reshape(7,9)
        X_m_c[i,:,:] = X_m
    
    return X_m_c

def make_matrix_8by1(X):
    
    [m,n] = np.shape(X)
    X_m_c = np.zeros([m,8,1])
    for i in range(m):
        X_m = X[i,:].reshape(8,1)
        X_m_c[i,:,:] = X_m
    
    return X_m_c