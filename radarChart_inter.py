# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:57:12 2019

@author: binbi
"""

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import glob
import os
import numpy as np
from plot import bar_accuracy
import matplotlib.style as style


style.use('ggplot')

"""
load accuracy of each speed for the previous 5 subjects


"""







filePaths = ("D:\\data_aquisition\\gunnar","D:\\data_aquisition\\hui","D:\\data_aquisition\\yazu_05_08","D:\\data_aquisition\\longbin","D:\\data_aquisition\\song","D:\\data_aquisition\\hao","D:\\data_aquisition\\yue","D:\\data_aquisition\\qing",)

df = pd.DataFrame()

      
for filePath in filePaths:

    os.chdir(filePath)
    
    filenames = glob.glob("CNN_1Con_0drop_0pool_1FC_512_100epo_*_intersubject_acc.csv")
    
    # read accuracy from collumns into rows
    
    list_of_dfs = [pd.read_csv(filename,header=None).T for filename in filenames]
    
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
     
    df =df.append(combined_df)
    
    #df.append(list_of_dfs)
    
    #df = df.add(combined_df)
    

df.columns= ['Overall', 'LR', 'MS', 'TS', 'PSw','Sw']





df_mean = df.mean()
df_std = df.std()



d = {'mean' : df_mean, 
      'std' : df_std} 

df_bar = pd.DataFrame(d) 


fig, ax = plt.subplots()
df_mean.plot.bar(yerr=df_std, ax=ax, capsize=4)
ax.set_ylabel('Acuracy',fontsize=15)
ax.set_ylim(ymin=0.6)
ax.set_xticklabels(('Overall', 'LR', 'MS', 'TS', 'PSw','Sw'))
ax.tick_params(axis='both', which='major', labelsize=15)


#plt.savefig('C:\\Users\\binbi\\Desktop\\my_phd\\git\\gait-recognition-DCNN\\barPlot_8_inter.png',dpi=400,bbox_inches='tight')



#boxplot = df.boxplot()
#boxplot.set_ylabel(fontsize=15)
#boxplot.set_xlabel(fontsize=15)
#
#plt.savefig('C:\\Users\\binbi\\Desktop\\my_phd\\git\\gait-recognition-DCNN\\boxPlot_12_inter.png',dpi=400,bbox_inches='tight')
#





