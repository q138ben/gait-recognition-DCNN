"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars.
"""
import numpy as np
import matplotlib.pyplot as plt





#men_means= np.arange(5)
#knn = (25, 32, 34, 20, 25)

def bar_accuracy(CNN):
    

    ind = np.arange(len(CNN))  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ax.bar(ind, CNN, width, color='SkyBlue', label='CNN')

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acuracy')
#    ax.set_title('Scores by group and gender')
    ax.set_ylim(ymin=0.7)
    ax.set_xticks(ind)
    ax.set_xticklabels(('Overall', 'LR', 'MS', 'TS', 'PSw','Sw'))
    ax.legend(loc=0)
    
#    ax.set_ylim(0,1.2)
    

    
    plt.show()
