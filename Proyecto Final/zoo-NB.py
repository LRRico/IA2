from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def Accuracy(fp,fn,vp,vn):
    return (vp+vn)/(vp+vn+fp+fn)
def Precision(fp, vp):
    try:
        return vp/(vp+fp)
    except:
        return 0.0
def Sensitivity(vp, fn):
    try:
        return vp/(vp+fn)
    except:
        return 0.0
def Specificity(vn,fp):
    try:
        return vn/(vn+fp)
    except:
        return 0.0
def F1Score(fp, vp, fn):
    try:
        return 2*vp /(2*vp+fp+fn)
    except:
        return 0.0

def NaiveBayes(csv_info):
    x = csv_info.iloc[:, 1:csv_info.shape[1]-1].values
    y = csv_info.iloc[:, csv_info.shape[1]-1].values
        
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    
    gnb = GaussianNB()
    y_pred = gnb.fit(xtrain, ytrain).predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
     
    fp = cm.sum(axis=0) - np.diag(cm)  
    fn = cm.sum(axis=1) - np.diag(cm)
    vp = np.diag(cm)
    vn = cm.sum() - (fp + fn + vp)
    print("NAIVE BAYES")
    print("Accuracy",Accuracy(fp, fn, vp, vn))
    print("Precision",Precision(fp, vp))
    print("Sensitivity",Sensitivity(vp, fn))
    print("Specificity",Specificity(vn,fp))
    print("F1 Score",F1Score(fp, vp, fn))

csv_info = pd.read_csv("zoo.data", header=None)
NaiveBayes(csv_info)