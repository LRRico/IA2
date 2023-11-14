from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn import svm
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

def DataPart(csv_info):
    cut = int(np.floor(csv_info.shape[0] * 0.9))
    csv_tr = csv_info.iloc[:cut,:]
    csv_ts = csv_info.iloc[cut:,:]
    return csv_tr,csv_ts

def SVM(csv_tr,csv_ts):
    xtr = csv_tr.iloc[:, 0:csv_info.shape[1]-1].values
    ytr = csv_tr.iloc[:, csv_info.shape[1]-1].values

    mvs = svm.SVC()
    mvs.fit(xtr,ytr)

    xts = csv_ts.iloc[:, 0:csv_info.shape[1]-1].values
    yts = csv_ts.iloc[:, csv_info.shape[1]-1].values
    
    fp = fn = vp = vn = 0
    ypd = [0]*csv_ts.shape[0]
    for i in range(csv_ts.shape[0]):
        ypd[i] = mvs.predict([xts[i]])
    cm = confusion_matrix(yts, ypd)
    
    if cm.size==2*2:
        fp = cm[0][1] 
        fn = cm[1][0]
        vp = cm[0][0]
        vn = cm[1][1]
    else:
        fp = cm.sum(axis=0) - np.diag(cm)  
        fn = cm.sum(axis=1) - np.diag(cm)
        vp = np.diag(cm)
        vn = cm.sum() - (fp + fn + vp)
    print("Accuracy",Accuracy(fp, fn, vp, vn))
    print("Precision",Precision(fp, vp))
    print("Sensitivity",Sensitivity(vp, fn))
    print("Specificity",Specificity(vn,fp))
    print("F1 Score",F1Score(fp, vp, fn))

csv_info = pd.read_csv("SwedenInsurance.csv")
print("|-Seguros Suecos")
try:
    csv_tr,csv_ts = DataPart(csv_info)
    SVM(csv_tr,csv_ts)
except:
    print("Error: Incapaz de aplicar solucion")
csv_info = pd.read_csv("Diabetes.csv")
print("|-Estadisticas de Diabetes")
csv_tr,csv_ts = DataPart(csv_info)
SVM(csv_tr,csv_ts)
csv_info = pd.read_csv("WineQuality.csv")
print("|-Calidad del Vino")
csv_tr,csv_ts = DataPart(csv_info)
SVM(csv_tr,csv_ts)



