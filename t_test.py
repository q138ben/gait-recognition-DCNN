# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:25:07 2020

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

from scipy.stats import ttest_rel

import seaborn as sns
from statannot import add_stat_annotation

#style.use('ggplot')


  



"""
Remenber to perform a paired t-test other than independent t-test. In a paired t-test, the length of the compared
dataset must be the same. But in independent t-test, the length of the compared dataset is usually different.
"""


filePaths = ("D:\\data_aquisition\\gunnar",
            "D:\\data_aquisition\\hui","D:\\data_aquisition\\marcus",
            "D:\\data_aquisition\\yazu_05_08","D:\\data_aquisition\\yixing","D:\\data_aquisition\\longbin",
            "D:\\data_aquisition\\song","D:\\data_aquisition\\hao","D:\\data_aquisition\\yue","D:\\data_aquisition\\qing")

#filePaths = ("D:\\data_aquisition\\gunnar","D:\\data_aquisition\\marcus","D:\\data_aquisition\\yazu_05_08","D:\\data_aquisition\\longbin","D:\\data_aquisition\\song","D:\\data_aquisition\\hao","D:\\data_aquisition\\yue","D:\\data_aquisition\\qing",)

df_correction_false = pd.DataFrame()
df_correction_true = pd.DataFrame()
df_acc_6_10 = pd.DataFrame()
df_acc_1_5 = pd.DataFrame()

acc_all =[]

for i, filePath in enumerate(filePaths):
    os.chdir(filePath)
    if i < 5:
        filenames_1 = glob.glob("all_speed_accuracy_*.npy")
        acc = np.load(filenames_1[0])
        acc = pd.DataFrame(acc)
        df_acc_1_5 = df_acc_1_5.append(acc)

    else:
        filenames_2 = glob.glob("CNN_1Con_0drop_0pool_1FC_512_*_frontal_*_acc.csv")
        list_of_dfs = [pd.read_csv(filename,header=None).T for filename in filenames_2]
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        df_acc_6_10 = df_acc_6_10.append(combined_df)
        
  
df_acc = df_acc_1_5.append(df_acc_6_10)
#df_acc.columns= ['Overall', 'LR', 'MS', 'TS', 'PSw','Sw']

for filePath in filePaths:

    os.chdir(filePath)
    
    filenames = glob.glob("*_intra_1_acc.csv")
    
    # read accuracy from collumns into rows
    
    list_of_dfs = [pd.read_csv(filename,sep=",",header=None) for filename in filenames]

    
    correction_false = list_of_dfs[0]
    correction_true = list_of_dfs[1]
    
    df_correction_false = df_correction_false.append(correction_false)
    df_correction_true = df_correction_true.append(correction_true)
    
#df_correction_false.columns= ['Overall', 'LR', 'MS', 'TS', 'PSw','Sw']
#df_correction_true.columns= ['Overall', 'LR', 'MS', 'TS', 'PSw','Sw']

def calc_p_values(value_1, value_2):
    t_values = np.zeros((5,6))
    p_values = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            t_value, p_value = ttest_rel(value_1.loc[i][value_1.columns[j]], value_2.loc[i][value_2.columns[j]])
            t_values[i,j] = t_value
            p_values[i,j] = p_value
    t_values = pd.DataFrame(t_values)
    p_values = pd.DataFrame(p_values)
    p_values.columns= ['Overall', 'LR', 'MS', 'TS', 'PSw','Sw']
    p_values.insert(0, "Speeds", ["\u03BC-2\u03C3", "\u03BC-\u03C3", "\u03BC", "\u03BC+\u03C3","\u03BC+2\u03C3"])   
    return t_values, p_values
        
#t_values, p_values = calc_p_values(df_acc, df_correction_false)
#
#t_values_1, p_values_2 = calc_p_values(df_correction_false,df_correction_true)
#
#t_values_2,p_values_3 = calc_p_values(df_acc,df_correction_true)


def create_dictionary(value_1,value_2, value_3):
    df_dics = {'baseline':value_1, 'feature_reduced':value_2 , 'feature_reduced_correction': value_3}
    gait_phase = ['Overall', 'LR', 'MS', 'TS', 'PSw','Sw']
    speeds = ["\u03BC-2\u03C3", "\u03BC-\u03C3", "\u03BC", "\u03BC+\u03C3","\u03BC+2\u03C3"]
    
    dics = pd.DataFrame()
    
    for key, value in df_dics.items():
        for i in range(5):
            for j in range(6):
                d =[{'acc': k, 'speed': speeds[i], 'gait_phase':gait_phase[j], 'mode': key} for k in value.loc[i][value.columns[j]]]
                # convert a list of dictionaries to dataframe
                d= pd.DataFrame(d)
                dics = dics.append(d)
    return dics

d = create_dictionary(df_acc,df_correction_false,df_correction_true)

# select dataframe with baseline values
d_baseline = d[(d['mode']=='baseline')] 

'''
        Statistical test to run. Must be one of:
        - `Levene`
        - `Mann-Whitney`
        - `Mann-Whitney-gt`
        - `Mann-Whitney-ls`
        - `t-test_ind`
        - `t-test_welch`
        - `t-test_paired`
        - `Wilcoxon`
        - `Kruskal`
'''

def plot_p_values(d,x,y,hue, test = 'Wilcoxon', save_figure = False):

    if x== "mode":
        box_pairs = [
        (("baseline","Overall"),("feature_reduced","Overall")),
        (("feature_reduced","Overall"),("feature_reduced_correction","Overall")),
        (("baseline","Overall"),("feature_reduced_correction","Overall"))
        ]
    if x == "speed":
        d = d[(d['mode']=='baseline')] 
        box_pairs = [
                    (("\u03BC-2\u03C3","Overall"),("\u03BC-\u03C3","Overall")),
                    (("\u03BC-2\u03C3","Overall"),("\u03BC","Overall")),
                    (("\u03BC-2\u03C3","Overall"),("\u03BC+\u03C3","Overall")),
                    (("\u03BC-2\u03C3","Overall"),("\u03BC+2\u03C3","Overall")),
                    (("\u03BC-\u03C3","Overall"),("\u03BC","Overall")),
                    (("\u03BC-\u03C3","Overall"),("\u03BC+\u03C3","Overall")),
                    (("\u03BC-\u03C3","Overall"),("\u03BC+2\u03C3","Overall")),
                    (("\u03BC","Overall"),("\u03BC+\u03C3","Overall")),
                    (("\u03BC","Overall"),("\u03BC+2\u03C3","Overall")),
                    (("\u03BC+\u03C3","Overall"),("\u03BC+2\u03C3","Overall")),                   
                    ]
    
    ax = sns.boxplot(data=d, x=x, y=y, hue=hue)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("Speeds",fontsize=14)
    ax.set_ylabel("Accuracy",fontsize=14)
    add_stat_annotation(ax, data=d, x=x, y=y, hue=hue, box_pairs=box_pairs,
                        test=test,comparisons_correction=None, loc='outside', verbose=2)
    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    if save_figure is True:
        os.chdir('C:\\Users\\binbi\\Desktop\\my_phd\\git\\gait-recognition-DCNN')
        plt.savefig('%s_p_value.png'%x, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

# plot p value between 3 different modes
plot_p_values(d,x= "mode",y = "acc", hue="gait_phase",save_figure=False)
# plot p value between speeds under baseline
plot_p_values(d,x= "speed",y = "acc", hue="gait_phase",save_figure= False)



#d_baseline['acc'].plot.hist(bins=50)
#d_baseline.skew(axis = 0, skipna = True)


