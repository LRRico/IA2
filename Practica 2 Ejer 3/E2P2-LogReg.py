from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
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

def LogicR(csv_info):
    x = csv_info.iloc[:, 0:csv_info.shape[1]-1].values
    y = csv_info.iloc[:, csv_info.shape[1]-1].values
            
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    
    sc_x = StandardScaler() 
    xtrain = sc_x.fit_transform(xtrain) 
    xtest = sc_x.transform(xtest) 
    
    classifier = LogisticRegression(random_state = 0) 
    classifier.fit(xtrain, ytrain)
    
    y_pred = classifier.predict(xtest)
    
    cm = confusion_matrix(ytest, y_pred)
     
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
    LogicR(csv_info)
except:
    print("Error: Incapaz de aplicar solucion")
csv_info = pd.read_csv("Diabetes.csv")
print("|-Estadisticas de Diabetes")
LogicR(csv_info)
csv_info = pd.read_csv("WineQuality.csv")
print("|-Calidad del Vino")
LogicR(csv_info)