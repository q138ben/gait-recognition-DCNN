# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:39:36 2018

@author: binbi
"""
import numpy as np


 

def phase_reorder(data,y_pred):
    
    phase_len = data.shape[0]
    phase_1 = []
    phase_2 = []
    phase_3 = []
    phase_4 = []
    phase_5 = []
    phase_1_pred = []
    phase_2_pred = []
    phase_3_pred = []
    phase_4_pred = []
    phase_5_pred = []
                
    

    for i in range(phase_len):
        if data[i] == 0:
            phase_1 = np.append(phase_1,data[i])
            phase_1_pred = np.append(phase_1_pred,y_pred[i])
            
        elif data[i] == 1:
            phase_2 = np.append(phase_2,data[i]) 
            phase_2_pred = np.append(phase_2_pred,y_pred[i])
        elif data[i] == 2:
            phase_3 = np.append(phase_3,data[i]) 
            phase_3_pred = np.append(phase_3_pred,y_pred[i])
        elif data[i] == 3:
            phase_4 = np.append(phase_4,data[i])
            phase_4_pred = np.append(phase_4_pred,y_pred[i])
        elif data[i] == 4:
            phase_5 = np.append(phase_5,data[i])
            phase_5_pred = np.append(phase_5_pred,y_pred[i])
        
        
    return phase_1, phase_2, phase_3, phase_4,phase_5,phase_1_pred,phase_2_pred,phase_3_pred,phase_4_pred,phase_5_pred
        
        