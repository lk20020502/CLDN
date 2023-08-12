#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;


def otherPerformance(y_true, y_pred,prt=True):
    F1=f1_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred)
    if prt:
        print("Pre: %.4f, Rec: %.4f, F1_score: %.4f" % (precision,recall,F1))
    return precision,recall,F1



