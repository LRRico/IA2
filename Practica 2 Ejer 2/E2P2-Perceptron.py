from sklearn import preprocessing
import pandas as pd
import random
import numpy as np

def Perceptron(csv_in):
    csv_info = preprocessing.normalize(csv_in)
    cut = int(np.floor(csv_info.shape[0] * 0.8))
    csv_t =  np.split(csv_info, [cut])
    csv_tr = csv_t[0]
    csv_ts = csv_t[1]
    
    max_epoch = 100
    lrn_rate = 0.8
    
    w = [0]*(csv_tr.shape[1]-1)
    for i in range(csv_tr.shape[1]-1):
        w[i]= random.uniform(-1,1)
    wY = random.uniform(-1,1)
    
    xY = 1
    
    pred = 1
    per_sum = 0
    
    for i in range(max_epoch):
        for x in range(csv_tr.shape[0]):
            row_v = csv_tr[x];
            
            for j in range(csv_tr.shape[1]-1):
                per_sum += row_v[j]*w[j]
            per_sum = xY*wY
            
            pred = 1 if per_sum >= 0 else -1
            
            for j in range(csv_tr.shape[1]-1):
                w[j] = w[j] +(lrn_rate * row_v[csv_tr.shape[1]-1]-pred * row_v[j])
            wY = wY + (lrn_rate * row_v[csv_tr.shape[1]-1]-pred * xY)
            
    fail = 0
    win = 0
    
    for i in range(csv_ts.shape[0]):
        row_o = csv_ts[i]
        
        for j in range(csv_tr.shape[1]-1):
            per_sum += row_v[j]*w[j]
        per_sum = xY*wY
            
        pred = 1 if per_sum >= 0 else -1
        
        if np.abs(pred-row_o[1])>0.5:
            fail += 1
        else:
            win += 1
    
    print("Resultados correctamente estimados",win)
    print("Resultados incorrectamente estimados",fail)
    
csv_in = pd.read_csv("Diabetes.csv")
   
csv_info = pd.read_csv("SwedenInsurance.csv")
print("Suecos")
Perceptron(csv_info)
csv_info = pd.read_csv("Diabetes.csv")
print("Diabetes")
Perceptron(csv_info)
csv_info = pd.read_csv("WineQuality.csv")
print("Vino")
Perceptron(csv_info)