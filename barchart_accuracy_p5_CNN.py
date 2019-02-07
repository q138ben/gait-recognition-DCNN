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
    rects1 = ax.bar(ind - width/2, CNN, width, 
                    color='SkyBlue', label='CNN')

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acuracy')
#    ax.set_title('Scores by group and gender')
    ax.set_ylim(ymin=0.7)
    ax.set_xticks(ind)
    ax.set_xticklabels(('Overall', 'LR', 'MS', 'TS', 'PSw','Sw'))
    ax.legend(loc='lower right')
    
#    ax.set_ylim(0,1.2)
    
    
    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.
    
        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """
    
        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')
    
    
#    autolabel(rects1, "left")
#    autolabel(rects2, "right")
    
    plt.show()
