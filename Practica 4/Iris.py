import numpy as np
import math
import pandas as pd

def BackP(csv_t, w,lr):
    n = csv_t.shape[0]
    n_con = len(w)
    
    for i in range(n):
        y = [0]*(n_con)
        e = [0]*(n_con)
        delta = [0]*(n_con)
        v = [0]*(n_con)
        x = np.random.rand(4,1)
        x[0] = csv_t.loc[i,0]
        x[1] = csv_t.loc[i,1]
        x[2] = csv_t.loc[i,2]
        x[3] = csv_t.loc[i,3]
        d = np.random.rand(3,1)
        d[0] = csv_t.loc[i,4]
        d[1] = csv_t.loc[i,5]
        d[2] = csv_t.loc[i,6]
        
        for j in range(n_con):
            if j == 0:
                v[j]=np.dot(w[j],x)
            else:
                v[j]=np.dot(w[j],y[j-1])
            if j == n_con-1:
                y[j]=Sigm(v[j])
            else:
                y[j]=Relu(v[j])
        for k in range(n_con-1, -1,-1):
            if k == n_con-1:
                e[k] = d - y[k]
                delta[k] = DerSigm(v[k])*e[k]
                dWl = w[k]+(lr*delta[k]*y[k-1])
            elif k == 0:
                e[k] = np.dot(w[k+1].T,delta[k+1])
                delta[k] = DerRelu(v[k])*e[k]
                dWl = w[k]+(lr*delta[k]*x.T)
            else:
                e[k] = np.dot(w[k+1].T,delta[k+1])
                delta[k] = DerRelu(v[k])*e[k]
                dWl = w[k]+(lr*delta[k]*y[k-1])
            w[k] = dWl
    return w

def Relu(x):
    g = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] > 0:
                g[i,j] = x[i,j]
            else:
                g[i,j] = 0
    return g

def DerRelu(x):
    g = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] > 0:
                g[i,j] = 1
            else:
                g[i,j] = 0
    return g

def Sigm(x):
    g = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            g[i,j] = 1 / (1 + np.exp(-x[i,j]))
    return g

def DerSigm(x):
    g = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            g[i,j] = np.exp(-x[i,j]) / ((1 + np.exp(-x[i,j]))**2)
    return g

def Softmax(x):
    x = x.copy()
    g = np.exp(x) / np.sum(np.exp(x), axis = 0)
    return g

def entr_y_prue(epoch, pesos,csv_tr,csv_ts):
    learn_r = 0.01
    for i in range(epoch):
        pesos = BackP(csv_tr,pesos, learn_r)
    total=0
    correcto=0
    corarray=[0]*(csv_ts[0].index.stop-csv_ts[0].index.start)
    n_con = len(pesos)
    for i in range(csv_ts[0].index.start,csv_ts[0].index.stop,1):
        y = [0]*(len(pesos))
        x = np.random.rand(4,1)
        v = [0]*(n_con)
        x[0] = csv_ts.loc[i,0]
        x[1] = csv_ts.loc[i,1]
        x[2] = csv_ts.loc[i,2]
        x[3] = csv_ts.loc[i,3]
        d = np.random.rand(3,1)
        d[0] = csv_ts.loc[i,4]
        d[1] = csv_ts.loc[i,5]
        d[2] = csv_ts.loc[i,6]
        for j in range(n_con):
            if j == 0:
                v[j]=np.dot(pesos[j],x)
            else:
                v[j]=np.dot(pesos[j],y[j-1])
            if j == n_con-1:
                y[j]=Sigm(v[j])
            else:
                y[j]=Relu(v[j])
        smy=Softmax(y[j])
        idy=np.argmax(smy)
        idd=np.argmax(d)
        if idy == idd:
            correcto +=1
        total +=1
        corarray[i]=smy[idy]
    return correcto/total,corarray

csv_info = pd.read_csv("irisbin.csv", header=None)

capas_ocult = 5
neuron_con = 3
epoch = 200
print("Capas Ocultas:",capas_ocult," | Neuronas por capa oculta:",neuron_con," | Epocas:",epoch)
if capas_ocult < 1:
    capas_ocult = 1
    
if neuron_con < 1:
    neuron_con = 1;

nt_pesos = (capas_ocult - 1) + 2
pesos = [0] * nt_pesos
for i in range(nt_pesos):
    if i == 0:
        pesos[i] = 2 * np.random.rand(neuron_con,4) -1 
    elif i == nt_pesos-1:
        pesos[i] = 2 * np.random.rand(3,neuron_con) -1
    else:
        pesos[i] = 2 * np.random.rand(neuron_con,neuron_con) -1
correcto = 0
tcar = []
#loo
for i in range(csv_info.shape[0]):
    if i == 0:
        csv_tss = csv_info.iloc[i,:]
        csv_tr = csv_info.iloc[i+1:,:]
        csv_tr = pd.concat([csv_tr], ignore_index=True, sort=False)
        csv_ts = pd.DataFrame(np.random.rand(1, 7))
        for j in range(csv_tss.shape[0]):
            csv_ts[j]=csv_tss[j]
    else:
        csv_tr1 = csv_info.iloc[:i,:]
        csv_tss = csv_info.iloc[i,:]
        csv_tr2 = csv_info.iloc[i+1:,:]
        csv_tr = pd.concat([csv_tr1,csv_tr2], ignore_index=True, sort=False)
        csv_ts = pd.DataFrame(np.random.rand(1, 7))
        for j in range(csv_tss.shape[0]):
            csv_ts[j]=csv_tss[j]
    cortol, car = entr_y_prue(epoch, pesos.copy(),csv_tr,csv_ts)
    correcto += cortol
    tcar += car
print("Leave One Out")
print("Tasa de exito:",(correcto/csv_info.shape[0])*100,"%")
print("Desviacion Estandar:", np.std(tcar))
print(" - - - - -")
#kf
correcto = 0
k=5
tcar = []
sec=math.floor(csv_info.shape[0]*(1/k))
for i in range(k):
    if i == 0:
        csv_ts = csv_info.iloc[:(i+1)*sec,:]
        csv_tr = csv_info.iloc[((i+1)*sec):,:]
        csv_ts = pd.concat([csv_ts], ignore_index=True, sort=False)
        csv_tr = pd.concat([csv_tr], ignore_index=True, sort=False)
    else:
        csv_tr1 = csv_info.iloc[:(i)*sec,:]
        csv_ts = csv_info.iloc[(i)*sec:(i)*sec + sec,:]
        csv_tr2 = csv_info.iloc[(i)*sec + sec:,:]
        csv_ts = pd.concat([csv_ts], ignore_index=True, sort=False)
        csv_tr = pd.concat([csv_tr1,csv_tr2], ignore_index=True, sort=False)
    cortol, car =  entr_y_prue(epoch, pesos.copy(),csv_tr,csv_ts)
    correcto += cortol
    tcar += car
print("K-Fold")
print("Tasa de exito:",(correcto/k)*100,"%")
print("Desviacion Estandar:", np.std(tcar))