from sklearn import svm
import pandas as pd
import numpy as np

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
    corr = 0
    incr = 0
    for i in range(csv_ts.shape[0]):
        o = mvs.predict([xts[i]])
        if o == yts[i]:
            corr +=1
        else:
            incr +=1
    print("Datos Correctamente Asignados",corr)
    print("Datos Incorrectamente Asignados",incr)

print("Diabetes")
csv_info = pd.read_csv("Diabetes.csv")
csv_tr,csv_ts = DataPart(csv_info)
SVM(csv_tr,csv_ts)
print("Sueco")
#csv_info = pd.read_csv("SwedenInsurance.csv")
#csv_tr,csv_ts = DataPart(csv_info)
#SVM(csv_tr,csv_ts)
print("Vino")
csv_info = pd.read_csv("WineQuality.csv")
csv_tr,csv_ts = DataPart(csv_info)
SVM(csv_tr,csv_ts)



