# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:39:15 2019

@author: binbi
"""

import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle


from sklearn.metrics import roc_curve, auc

from scipy import interp
import itertools
import matplotlib.style as style
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes,rotation=45)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True phase')
    plt.xlabel('Predicted phase')
    plt.tight_layout()



def plot_roc(model,n_classes,X_test,y_test):
    
    y_score = model.predict_proba(X_test)

    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    ##############################################################################
    # Plot of a ROC curve for a specific class

    lw = 2
    
    
    ##############################################################################
    # Plot ROC curves for the multiclass problem
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green'])
    phases = cycle(['LR','MS','TS','PSw','Sw'])
    for i, color,phase in zip(range(n_classes), colors,phases):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(phase, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()






def plot_loss_and_acc(history):
    
    style.use('ggplot')
    
    # summarize history for accuracy
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('DCNN Model Accuracy')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')
#    plt.legend(['Training data', 'Validation data'], loc='best')
#    plt.show()
#    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('DCNN Model Loss')
#    plt.ylabel('Loss')
#    plt.xlabel('Epoch')
#    plt.legend(['Training data', 'Validation data'], loc='best')
#    plt.grid()
#    plt.show()
    
    fig,(ax1,ax2)= plt.subplots(2,1,figsize=(4,5))
    ax1.plot(history.history['acc'],label='Traing data')
    ax1.plot(history.history['val_acc'],label='Test data')
    ax1.legend(loc='best')
    ax1.set_ylabel('Accuracy')
#    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticklabels([])
#    ax1.xaxis.set_major_formatter(plt.NullFormatter())
#    ax1.set_legend(['Training data', 'Validation data'], loc='best')
    
    ax2.plot(history.history['loss'],label='Traing data')
    ax2.plot(history.history['val_loss'],label='Test data')
    ax2.legend(loc='best')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
#    ax2.set_legend(['Training data', 'Validation data'], loc='best')

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

def bar_accuracy(CNN,std):
    style.use('ggplot')
    
    ind = np.arange(len(CNN))  # the x locations for the groups
    width = 0.5  # the width of the bars
    
#    std=[0.01,0.02,0.02,0.02,0.02,0]
    fig, ax = plt.subplots()
    ax.bar(ind, CNN, width,yerr=std,color='SkyBlue', align='center',alpha=0.8, ecolor='black',capsize=2)
    
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acuracy',fontsize=15)
#    ax.set_title('CNN recognition accuracy on 5 gait phases',fontsize=15)
    ax.set_ylim(ymin=0.7)
    ax.set_xticks(ind)
    ax.set_xticklabels(('Overall', 'LR', 'MS', 'TS', 'PSw','Sw'))
    ax.tick_params(axis='both', which='major', labelsize=12)
#    ax.legend(loc=0,fontsize=15)
    
#    ax.set_ylim(0,1.2)
    plt.savefig('gunnar_bar.png',dpi=400)   
    plt.show()

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
        

def combineImage():
    img1 = mpimg.imread('hui_bar.png')
    img2 = mpimg.imread('marcus_bar.png')
    img3 = mpimg.imread('gunnar_bar.png')
    img4 = mpimg.imread('yanzu_bar.png')
    img5 = mpimg.imread('yixing_bar.png')
    img6 = mpimg.imread('snorri_bar.png')
    
    fig,[[ax1,ax2],[ax3,ax4],[ax5,ax6]] = plt.subplots(3,2)
    
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(img3)
    ax4.imshow(img4)
    ax5.imshow(img5)
    ax6.imshow(img6)
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    
    plt.savefig('all_bar',dpi=400)
    
    plt.tight_layout()

    plt.show()



#bar_accuracy(mean,std)

#combineImage()
