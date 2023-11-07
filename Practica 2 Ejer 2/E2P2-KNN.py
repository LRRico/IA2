import pandas as pd
import numpy as np

def DataPart(csv):
    cut = int(np.floor(csv_info.shape[0] * 0.8))
    tr = csv_info.iloc[:cut,:]
    ts = csv_info.iloc[cut:,:]
    return tr,ts
def KNN(csv_tr,csv_ts,nn,tipo):
    e = 0
    for h in range(csv_ts.shape[0]):
        nl = np.array([1000.000]*nn)
        nr = np.array([0]*nn)
        a = csv_ts.iloc[h]
        for i in range(csv_tr.shape[0]):
            tsum = 0
            for j in range(csv_tr.shape[1]-1):
                tsum += (csv_tr.iloc[i][j] - a[j])**2
            dist = tsum**0.5
            if(nl[nl.argmax()]>dist):
                nl[nl.argmax()] = dist
                nr[nl.argmax()] = csv_tr.iloc[i][csv_tr.shape[1]-1]
        if tipo:
            prom = np.mean(nr)
            e += np.abs(prom - a[csv_tr.shape[1]-1])
        else:
            rf = np.argmax(np.bincount(nr))
            if(rf!=a[csv_tr.shape[1]-1]):
                e+=1
    if tipo:
        print("Valor promedio de error",e/h)
    else:
        print("Porcentaje de error:",e/h)
print("|-Seguros Suecos")
csv_info = pd.read_csv("SwedenInsurance.csv")
csv_tr,csv_ts = DataPart(csv_info)
neighbor_n = 4
KNN(csv_tr,csv_ts,neighbor_n,1)

print("|-Estadisticas de Diabetes")
csv_info = pd.read_csv("Diabetes.csv")
csv_tr,csv_ts = DataPart(csv_info)
neighbor_n = 3
KNN(csv_tr,csv_ts,neighbor_n,0)

print("|-Calidad del Vino")
csv_info = pd.read_csv("WineQuality.csv")
csv_tr,csv_ts = DataPart(csv_info)
neighbor_n = 3
KNN(csv_tr,csv_ts,neighbor_n,0)