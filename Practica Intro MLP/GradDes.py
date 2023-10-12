import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random as rd

### FUNCION PARA CREAR GRAFICOS
def superficie(f,x):
    # Decl.Rango de visualizacion de las graficas
    xmin = [-3, -3]
    xmax = [3, 3]
    # Preparacion de la superficie de la grafica segun la formula
    X = np.arange(xmin[0], xmax[0], 0.25)
    Y = np.arange(xmin[1], xmax[1], 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    # Declaracion de figura (Espacio 3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Insertar superficie en figura
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
    # Datos explicativos de la figura
    ax.set_title('Superficie')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Insertar punto optimo encontrado e insertar leyenda
    ax.scatter(x[0], x[1], f(x[0],x[1]), c='r', label='Óptimo', s=120)
    ax.legend()
    # Mostrar grafico
    plt.show()
 
### DECLARACION DE VARIABLES
# Decl.Formula objetivo y derivada de la misma
form = lambda x1,x2: 10 - math.exp(-(x1**2 + 3*x2**2))
derv = lambda x1,x2: -( math.exp(-(x1**2 + 3*x2**2))) * (-2*x1 - 6*x2)
# Decl.Punto Optimo (ubicacion inicial)
xi = [rd.uniform(-1,1), rd.uniform(-1,1)]
xi = np.asanyarray(xi)
# Decl.Epocas
epoch = 500
# Decl.Tasa de aprendizaje
trn = 0.05

### GRADIENTE DESCENDIENTE
# Paso de Epocas
for i in range(epoch):              
    # Calculacion de derivada del punto actual
    dxi = derv(xi[0], xi[1])
    # Remplazo de Punto Optimo al restarle derivada por factor entrenamiento
    xi = xi - trn * dxi

### MOSTRAR DATOS
# Vectorizacion de la formula para graficar
form = np.vectorize(form)
# Mostrar datos(Graf Superficie, y Minimo Global)
superficie(form, xi)
print("Mínimo global en:", xi, "f(x) =",form(xi[0],xi[1]))