# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:12:14 2019

@author: binbi
"""

import numpy as np

from fetchDataset import fetchDataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


X1,y1= fetchDataset('song_frontal_5.0')

#X = X1[:,8:] 

LR_ind = [i for i, e in enumerate(y1) if e == 1]
MS_ind = [i for i, e in enumerate(y1) if e == 2]
TS_ind = [i for i, e in enumerate(y1) if e == 3]
PSw_ind = [i for i, e in enumerate(y1) if e ==4]
Sw_ind = [i for i, e in enumerate(y1) if e == 5]




LR_ind_diff = np.diff(LR_ind)
Sw_ind_diff = np.diff(Sw_ind)


gait_ind_start = np.array([LR_ind[i+1] for i, e in enumerate(LR_ind_diff) if e > 5] ) # get the index for the start of LR in each gait cycle

gait_ind_end = np.array([Sw_ind[i] for i, e in enumerate(Sw_ind_diff) if e > 5] )  # get the index for the end of Sw in each gait cycle


gait_ind_start = np.insert(gait_ind_start,0,0) # insert index 0 
gait_ind_start = np.delete(gait_ind_start,-1) # delete the  last index becaur it is not a full cycle


num_chan = np.shape(X1)[1]
gait_num = 9  # choose the gait cycle


tf = np.linspace(0,100,gait_ind_end[gait_num]-gait_ind_start[gait_num])


for i in range(63):

    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Gait cycle')
    #ax1.set_ylabel('Normalized angular velocity of \n right foot along y axis', color=color)
    ax1.set_ylabel('Normalized data', color=color)
    ax1.plot(tf,X1[gait_ind_start[gait_num]:gait_ind_end[gait_num],i], color=color) # plot the right foot angular velocity of y axis 57
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax1.set_title('chanel num ' + str(i))
