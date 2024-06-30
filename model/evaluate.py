import pickle
import numpy as np
from tqdm import tqdm
from itertools import chain
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
def cal_metrics(label_list, pred_list):
    labels = []
    labelb = []
    for label in label_list:
        label=label.squeeze(-1)
        label_idx = []
        for i in range(label.shape[0]):
            label_idx.append(int(label[i]))
        labels.append(label_idx)
    preds = []
    for pred in pred_list:
        pred=pred.squeeze(-1)
        pred_idx = []
        labelb_idx = []
        for i in range(pred.shape[0]):
            pred_idx.append(float(pred[i]))
            if(float(pred[i]>0.5)):
                labelb_idx.append(1)
            else:
                labelb_idx.append(0)
        preds.append(pred_idx)
        labelb.append(labelb_idx)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    labelb = np.concatenate(labelb)


    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels,labelb)
    f1 = f1_score(labels,labelb,average='binary')
    p = precision_score(labels, labelb, average='binary') 
    r = recall_score(labels, labelb, average='binary')
    #print('f1_micro: {0}'.format(f1_micro))
    #print('f1_macro: {0}'.format(f1_macro))
    # mean_absolute_error = mean_absolute_error(labels, preds)
    # mean_squared_error = mean_squared_error(labels, preds)
    rmse = sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)


    return auc,acc,f1,p,r,rmse,r2


def drawroc(label_list, pred_list,path,epoch,seed):

    labels = []
    labelb = []
    for label in label_list:
        label=label.squeeze(-1)
        label_idx = []
        for i in range(label.shape[0]):
            label_idx.append(int(label[i]))
        labels.append(label_idx)
    preds = []
    for pred in pred_list:
        pred=pred.squeeze(-1)
        pred_idx = []
        labelb_idx = []
        for i in range(pred.shape[0]):
            pred_idx.append(float(pred[i]))
            if(float(pred[i]>0.5)):
                labelb_idx.append(1)
            else:
                labelb_idx.append(0)
        preds.append(pred_idx)
        labelb.append(labelb_idx)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    fpr = dict()
    tpr = dict() 
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(labels, preds)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 2
    plt.plot(fpr[0], tpr[0],
         lw=lw, label= ' (area = %0.5f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    plt.title('ROC Curve')
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig(path+str(epoch)+'_'+str(seed)+"_roc.pdf")
    plt.clf()

def drawprc(label_list, pred_list,path,epoch,seed):
    
    labels = []
    labelb = []
    for label in label_list:
        label=label.squeeze(-1)
        label_idx = []
        for i in range(label.shape[0]):
            label_idx.append(int(label[i]))
        labels.append(label_idx)
    preds = []
    for pred in pred_list:
        pred=pred.squeeze(-1)
        pred_idx = []
        labelb_idx = []
        for i in range(pred.shape[0]):
            pred_idx.append(float(pred[i]))
            if(float(pred[i]>0.5)):
                labelb_idx.append(1)
            else:
                labelb_idx.append(0)
        preds.append(pred_idx)
        labelb.append(labelb_idx)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    lw = 2
    lr_precision, lr_recall, _ = precision_recall_curve(labels, preds)    
    #   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
    plt.plot(lr_recall, lr_precision, lw = 2, label= ' (area = %0.5f)' % accuracy_score(labels, preds))
    plt.plot([0, 1], [0, 1], color='navy', lw= lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(path+str(epoch)+'_'+str(seed)+"_prc.pdf")
    plt.clf()

