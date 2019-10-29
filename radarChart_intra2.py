# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:03:56 2019

@author: binbi
"""

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
laod accuracy of each speed for the last 5 subject
"""




filePaths = ("D:\\data_aquisition\\gunnar","D:\\data_aquisition\\marcus","D:\\data_aquisition\\yazu_05_08","D:\\data_aquisition\\longbin","D:\\data_aquisition\\song","D:\\data_aquisition\\hao","D:\\data_aquisition\\yue","D:\\data_aquisition\\qing",)

df = pd.DataFrame()

      
for filePath in filePaths:

    os.chdir(filePath)
    
    filenames = glob.glob("CNN_1Con_0drop_0pool_1FC_512_100epo_*_pooledSpeed_acc.csv")
    
    # read accuracy from collumns into rows
    
    list_of_dfs = [pd.read_csv(filename,header=None).T for filename in filenames]
    
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
     
    df =df.append(combined_df)
    
    #df.append(list_of_dfs)
    
    #df = df.add(combined_df)
    

df.columns= ['Overall', 'LR', 'MS', 'TS', 'PSw','Sw']
 
#df = df.add(df1)   
#
#
#df = df/10
#
#
df_mean = df.mean()
df_std = df.std()
#
#df.columns = ["Overall", "LR", "MS", "TS","PSw","Sw"]
#df.insert(0, "Speeds", ["\u03BC-2\u03C3", "\u03BC-\u03C3", "\u03BC", "\u03BC+\u03C3","\u03BC+2\u03C3"], True)   
## Set data
#
##df= pd.read_excel('intra1.xlsx', index_col=None)
#
#
# 
## ------- PART 1: Create background
# 
## number of variable
#categories=list(df)[1:]
#N = len(categories)
# 
## What will be the angle of each axis in the plot? (we divide the plot / number of variable)
#angles = [n / float(N) * 2 * pi for n in range(N)]
#angles += angles[:1]
# 
## Initialise the spider plot
#ax = plt.subplot(111, polar=True)
# 
## If you want the first axis to be on top:
#ax.set_theta_offset(pi / 2)
#ax.set_theta_direction(-1)
# 
## Draw one axe per variable + add labels labels yet
#plt.xticks(angles[:-1], categories,size = 15)
# 
## Draw ylabels
#ax.set_rlabel_position(0)
##plt.yticks([0.6,0.7,0.9], ["0.6","0.7","0.9"], color="black", size=10)
##plt.ylim(0.6,1)
# 
#plt.yticks([0.89,0.93,0.97], ["0.89","0.93","0.97"], color="black", size=10)
#plt.ylim(0.85,1)
#
#
#
## ------- PART 2: Add plots
# 
## Plot each individual = each line of the data
## I don't do a loop, because plotting more than 3 groups makes the chart unreadable
# 
## Ind1
#
#for i in range(5):
#    
#    values=df.loc[i].drop('Speeds').values.flatten().tolist()
#    values += values[:1]
#    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.loc[i].Speeds)
##    ax.fill(angles, values, 'b', alpha=0.1)
#     
#    # Ind2
##    values=df.loc[2].drop('Subjects').values.flatten().tolist()
##    values += values[:1]
##    ax.plot(angles, values, linewidth=1, linestyle='solid', label="group B")
##    ax.fill(angles, values, 'r', alpha=0.1)
# 
## Add legend
#plt.legend(loc='lower left',bbox_to_anchor=(1.1, 0))
#
#plt.show
#
#plt.savefig('spiderPlot_10_intra.png',dpi = 400)
#
#
d = {'mean' : df_mean, 
      'std' : df_std} 

df_bar = pd.DataFrame(d) 


fig, ax = plt.subplots()
df_mean.plot.bar(yerr=df_std, ax=ax, capsize=4)
ax.set_ylabel('Acuracy',fontsize=15)
ax.set_ylim(ymin=0.8)
ax.set_xticklabels(('Overall', 'LR', 'MS', 'TS', 'PSw','Sw'))
ax.tick_params(axis='both', which='major', labelsize=15)
#
#
#plt.savefig('C:\\Users\\binbi\\Desktop\\my_phd\\git\\gait-recognition-DCNN\\barPlot_8_intra2.png',dpi=400,bbox_inches='tight')





  




