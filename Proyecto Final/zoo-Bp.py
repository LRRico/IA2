import numpy as np
import math
import pandas as pd


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


def BackP(csv_t, w,lr):
    n = csv_t.shape[0]
    n_con = len(w)
    
    for i in range(n):
        y = [0]*(n_con)
        e = [0]*(n_con)
        delta = [0]*(n_con)
        v = [0]*(n_con)
        x = np.random.rand(16,1)
        for j in range(16):
            x[j] = csv_t.iloc[i,j]
        d = np.zeros((7,1))
        d[csv_t.loc[i,17]-1] = 1
        
        for j in range(n_con):
            if j == 0:
                v[j]=np.dot(w[j],x)
            else:
                v[j]=np.dot(w[j],y[j-1])
            if j == n_con-1:
                y[j]=Softmax(v[j])
            else:
                y[j]=Sigm(v[j])
                
        for k in range(n_con-1, -1,-1):
            if k == n_con-1:
                e[k] = d - y[k]
                delta[k] = e[k]
                dWl = w[k] + (lr*np.dot(delta[k],y[k-1].T))
            elif k == 0:
                e[k] = np.dot(w[k+1].T,delta[k+1])
                delta[k] = DerSigm(v[k])*e[k]
                dWl = w[k] + (lr*delta[k]*x.T)
            else:
                e[k] = np.dot(w[k+1].T,delta[k+1])
                delta[k] = DerSigm(v[k])*e[k]
                dWl = w[k] + (lr*delta[k]*y[k-1].T)
            w[k] = dWl
    return w

def DerSoftmax(x):
    g = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] > 0:
                g[i,j] = 1
            else:
                g[i,j] = 0
    return g

def Softmax(x):
    ex = np.exp(x.copy())
    sumex = np.sum(ex)
    return ex/sumex

def DerSigm(x):
    g = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            g[i,j] = math.exp(-x[i,j]) / ((1 + math.exp(-x[i,j]))**2)
    return g

def Sigm(x):
    g = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            g[i,j] = 1 / (1 + math.exp(-x[i,j]))
    return g

def entr_y_prue(epoch, pesos,csv_tr,csv_ts,cm):
    learn_r = 0.9
    for i in range(epoch):
        pesos = BackP(csv_tr,pesos, learn_r)
    n_con = len(pesos)
    for i in range(csv_ts.index.start,csv_ts.index.stop,1):
        y = [0]*(len(pesos))
        v = [0]*(n_con)
        x = np.random.rand(16,1)
        for j in range(16):
            x[j] = csv_ts.iloc[i,j]
        d = np.zeros((7,1))
        d[csv_ts.loc[i,17]-1] = 1
        for j in range(n_con):
            if j == 0:
                v[j]=np.dot(pesos[j],x)
            else:
                v[j]=np.dot(pesos[j],y[j-1])
            y[j]=Sigm(v[j])
        cm[np.argmax(d),np.argmax(y[j])] += 1
    return cm

csv_info = pd.read_csv("zoo.data", header=None)

capas_ocult = 1
neuron_con = 10
epoch = 100

if capas_ocult < 1:
    capas_ocult = 1
    
if neuron_con < 1:
    neuron_con = 1;

cm = np.zeros((7,7))

nt_pesos = (capas_ocult - 1) + 2
pesos = [0] * nt_pesos
for i in range(nt_pesos):
    if i == 0:
        pesos[i] = 2 * np.random.rand(neuron_con,csv_info.shape[1]-2) -1
    elif i == nt_pesos-1:
        pesos[i] = 2 * np.random.rand(7,neuron_con) -1
    else:
        pesos[i] = 2 * np.random.rand(neuron_con,neuron_con) -1

k=5
sec=math.floor(csv_info.shape[0]*(1/k))
for i in range(k):
    if i == 0:
        csv_ts = csv_info.iloc[:(i+1)*sec,1:]
        csv_tr = csv_info.iloc[((i+1)*sec):,1:]
        csv_ts = pd.concat([csv_ts], ignore_index=True, sort=False)
        csv_tr = pd.concat([csv_tr], ignore_index=True, sort=False)
    else:
        csv_tr1 = csv_info.iloc[:(i)*sec,1:]
        csv_ts = csv_info.iloc[(i)*sec:(i)*sec + sec,1:]
        csv_tr2 = csv_info.iloc[(i)*sec + sec:,1:]
        csv_ts = pd.concat([csv_ts], ignore_index=True, sort=False)
        csv_tr = pd.concat([csv_tr1,csv_tr2], ignore_index=True, sort=False)
    cm = entr_y_prue(epoch, pesos.copy(),csv_tr,csv_ts,cm)

fp = cm.sum(axis=0) - np.diag(cm)  
fn = cm.sum(axis=1) - np.diag(cm)
vp = np.diag(cm)
vn = cm.sum() - (fp + fn + vp)
print("RED NEURONAL")
print("Accuracy",Accuracy(fp, fn, vp, vn))
print("Precision",Precision(fp, vp))
print("Sensitivity",Sensitivity(vp, fn))
print("Specificity",Specificity(vn,fp))
print("F1 Score",F1Score(fp, vp, fn))
