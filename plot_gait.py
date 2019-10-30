# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:12:14 2019

@author: binbi
"""

import numpy as np

from fetchDataset import fetchDataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.style as style





## plot all channels data in a certain gait defined by gait_num

def plot_all_chan(X1,gait_ind_start,gait_ind_end,gait_num):
    
    style.use('ggplot')
    
    # geit how many channels in total in the data
    num_chan = np.shape(X1)[1]
    
    
    tf = np.linspace(0,100,gait_ind_end[gait_num]-gait_ind_start[gait_num])
    
    for i in range(num_chan):
    
        fig, ax1 = plt.subplots()
        

        ax1.set_xlabel('Gait cycle')
        #ax1.set_ylabel('Normalized angular velocity of \n right foot along y axis', color=color)
        ax1.set_ylabel('Normalized data')
        ax1.plot(tf,X1[gait_ind_start[gait_num]:gait_ind_end[gait_num],i]) # plot the right foot angular velocity of y axis 57
        ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax1.set_title('chanel num ' + str(i))
    


# plot a specific channel in many gait cycle

def plot_chan_mul_cyl(X1,gait_ind_start,gait_ind_end, chan_num,total_gait_num):
    
    style.use('ggplot')

    # to plot multiple plot on one figure, the plt.subplots() must be outside the loop
    fig, ax1 = plt.subplots()
       
    for i in range(total_gait_num):
            
        tf = np.linspace(0,100,gait_ind_end[i]-gait_ind_start[i])
        
        
        ax1.set_xlabel('Gait cycle',fontsize=15)
        #ax1.set_ylabel('Normalized angular velocity of \n right foot along y axis', color=color)
        ax1.set_ylabel('Normalized data on \n channel ' + str(chan_num), fontsize=15)
        ax1.plot(tf,X1[gait_ind_start[i]:gait_ind_end[i],chan_num]) # plot the right foot angular velocity of y axis 57
        ax1.tick_params(labelsize=13)
        ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))





# plot specific channel on all gait cycle with mean and std

def plot_chan(X1,gait_ind_start,gait_ind_end, chan_num):

    # get how many gait cycle in the data
    num_gait = len(gait_ind_start)
    
    style.use('ggplot')
    
    df = pd.DataFrame()
    
    for i in range(num_gait):
        
        series = pd.Series(X1[gait_ind_start[i]:gait_ind_end[i],chan_num]) # creat dataframe series on every gait cycle on specific channel
        
        df =df.append(series,ignore_index=True)
    
    
    # means of all gait cycle on specific channel e.g.53. 
    df_mean = df.mean()
    df_std = df.std()
    
    
    tf = np.linspace(0,100,len(df_mean))
    
    fig, ax = plt.subplots()
    tf = np.linspace(0,100,len(df_mean))
    ax.plot(tf,df_mean)
    ax.fill_between(tf,df_mean-df_std, df_mean + df_std,color='b', alpha=0.2)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.set_xlabel('Gait cycle',fontsize=15)
    ax.set_ylabel('Normalized data on \n channel ' + str(chan_num), fontsize=15)
    ax.tick_params(labelsize=13)
    





X1,y1= fetchDataset('hui_5.5')
#X2,y2= fetchDataset('song_frontal_5.0')
#
#X1 = np.concatenate((X1,X2))
#y1 = np.concatenate((y1,y2))




X1 = X1[:,8:] 

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



"""
Plot

"""

plot_chan(X1,gait_ind_start,gait_ind_end, chan_num= 53)

#plot_chan_mul_cyl(X1,gait_ind_start,gait_ind_end, chan_num=49,total_gait_num = 5)


#plot_all_chan(X1,gait_ind_start,gait_ind_end,gait_num= 2)


    
    
