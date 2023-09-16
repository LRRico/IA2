import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import numpy as np

def Fgraph(xX,yX,zX,xF,yF,zF):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p1 = np.array([xX,yX,zF])
    p2 = np.array([xX,yF,zX])
    p3 = np.array([xF,yX,zX])

    u = p3 - p1
    v = p2 - p1

    u_v = np.cross(u, v)
    
    d = np.dot(u_v, p3)
    
    point  = np.array(p1)
    normal = np.array(u_v)
    
    d = -point.dot(normal)
    
    xx, yy = np.meshgrid(range(100), range(100))
    
    z = np.array((-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2])
    
    ax.plot_surface(xx, yy, z,alpha=0.2)
    
    ax.scatter(xX, yX, zF, c='g', label='Óptimo', s=100)
    ax.scatter(xX, yF, zX, c='g', label='Óptimo', s=100)
    ax.scatter(xF, yX, zX, c='g', label='Óptimo', s=100)
    
    ax.scatter(1, 1, 1, c='b', label='Óptimo', s=50)
    ax.scatter(1, 1, -1, c='b', label='Óptimo', s=50)
    ax.scatter(1, -1, 1, c='b', label='Óptimo', s=50)
    ax.scatter(1, -1, -1, c='b', label='Óptimo', s=50)
    ax.scatter(-1, 1, 1, c='b', label='Óptimo', s=50)
    ax.scatter(-1, 1, -1, c='b', label='Óptimo', s=50)
    ax.scatter(-1, -1, 1, c='b', label='Óptimo', s=50)
    ax.scatter(-1, -1, -1, c='b', label='Óptimo', s=50)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    plt.show(block=False)
    plt.pause(.05)

def training(csv_info,trn_p):
    cut = math.floor(csv_info.shape[0] * (trn_p))
    csv_in = csv_info.iloc[:cut,:]
    csv_out = csv_info.iloc[cut:,:]
  
    csv_in.reset_index(drop=True)
    csv_out.reset_index(drop=True)
    
    max_epoch = 50
    lrn_rate = 0.2
    
    w1 = random.uniform(-1,1)
    w2 = random.uniform(-1,1)
    w3 = random.uniform(-1,1)
    wY = random.uniform(-1,1)
    
    xY = 1
    
    pred = 1
    per_sum = 0
    
    print("Entrenamiento...")
    for i in range(max_epoch):
        for x in range(0,160):
            row_v = csv_in.iloc[x];
            
            per_sum = (row_v[0]*w1 + row_v[1]*w2 + row_v[2]*w3 + xY*wY)
            
            if per_sum >= 0:
                pred = 1
            else:
                pred = -1
                
            w1 = w1 + (lrn_rate * (row_v[3]-pred) * row_v[0])
            w2 = w2 + (lrn_rate * (row_v[3]-pred) * row_v[1])
            w3 = w3 + (lrn_rate * (row_v[3]-pred) * row_v[2])
            wY = wY + (lrn_rate * (row_v[3]-pred) * xY)
            
    print("Pesos Finales: W1=" ,"{:.4f}".format(w1) , " W2=", "{:.4f}".format(w2)," W3=","{:.4f}".format(w3)," WY=","{:.4f}".format(wY))
    print("Pruebas...")
    fail = 0
    win = 0
    
    for i in range(csv_out[0].index.start,csv_out[0].index.stop):
        row_o = csv_out.loc[i]
        per_sum = (row_o[0]*w1 + row_o[1]*w2 + row_o[2]*w3 + xY*wY)        
        pred = 1 if per_sum >= 0 else -1
        if pred != row_o[3]:
            fail += 1
        else:
            win += 1
    
    print("Resultados correctamente estimados",win)
    print("Resultados incorrectamente estimados",fail)
    print(" - - - - - - - -")
    
    Fgraph(row_o[0],row_o[1],row_o[2],w1,w2,w3)

def part_trn(met,trn_pc,csv_input):
    if met == 1:
        print("- Metodo Aleatorio")
        csv_result = csv_input.sample(replace=False,frac = 1,ignore_index=True)
    elif met == 2:
        print("- Metodo Division Lineal (Segun entrada)")
        csv_result = csv_input
    else:
        print("- Metodo Organizacion ( X",met-2,")")
        csv_result = csv_input.sort_values(csv_input.columns[met-3],axis=0, inplace=False,ignore_index=True)
    training(csv_result,trn_pc)

csv_input = pd.read_csv("spheres1d10.csv",header=None)

n_particiones = 5
trn_pc = 0.8

part_trn(1,trn_pc,csv_input)
part_trn(2,trn_pc,csv_input)
part_trn(3,trn_pc,csv_input)
part_trn(4,trn_pc,csv_input)
part_trn(5,trn_pc,csv_input)