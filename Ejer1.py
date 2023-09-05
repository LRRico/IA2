import pandas as pd
import random
import matplotlib.pyplot as plt

def Fgraph(xF1,yF1,xF2,yF2):
    plt.rcParams["figure.autolayout"] = True
    plt.grid()

    xP1 = [1];yP1 = [1]
    xP2 = [1];yP2 = [-1]
    xP3 = [-1];yP3 = [1]
    xP4 = [-1];yP4 = [-1]
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.plot([xF1,xF2],[yF1,yF2])
    
    plt.plot(xP1,yP1, marker="o", markersize=8, markerfacecolor="blue")
    plt.plot(xP2,yP2, marker="o", markersize=8, markerfacecolor="blue")
    plt.plot(xP3,yP3, marker="o", markersize=8, markerfacecolor="blue")
    plt.plot(xP4,yP4, marker="o", markersize=8, markerfacecolor="blue")

    plt.title("Grafico")
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show(block=False)
    plt.pause(.05)


csv_in = pd.read_csv("XOR_trn.csv",header=None)

max_epoch = 50
lrn_rate = 0.2

w1 = random.uniform(-1,1)
w2 = random.uniform(-1,1)
wY = random.uniform(-1,1)

xY = 0

pred = 1
per_sum = 0
print("Iniciando entrenamiento...")
for i in range(max_epoch):
    for x in range(csv_in.shape[0]):
        row_v = csv_in.loc[x];
        
        per_sum = (row_v[0]*w1 + row_v[1]*w2 + xY*wY)
        
        if per_sum >= 0:
            pred = 1
        else:
            pred = 0
            
        w1 = w1 + (lrn_rate * (row_v[2]-pred) * row_v[0])
        w2 = w2 + (lrn_rate * (row_v[2]-pred) * row_v[1])
        wY = wY + (lrn_rate * (row_v[2]-pred) * xY)
        
print("Entrenamiento terminado")
print("Pesos Finales:", w1 , " ", w2," ",wY)

print("Iniciando Pruebas...")

fail = 0
win = 0
csv_out = pd.read_csv("XOR_tst.csv",header=None)

for i in range(csv_out.shape[0]):
    row_o = csv_out.loc[i]
    
    per_sum = (row_o[0]*w1 + row_o[1]*w2 + xY*wY)
    
    pred = 1 if per_sum >= 0 else -1
    
    if pred != row_o[2]:
        fail += 1
    else:
        win += 1

print("Pruebas terminadas")
print("Resultados correctamente estimados",win)
print("Resultados incorrectamente estimados",fail)

Fgraph(row_o[0],w1,row_o[1],w2)
