from Funciones import *  
from Funciones_simbolico import *  

# Para limpiar terminal
import os 
os.system("clear") 

# Hacer copias
from copy import copy

# Realizar graficos
import matplotlib.pyplot as plt

# Realizar operaciones
import numpy as np

# Generalizar transf. homogenea T
d_vacio=np.array([[0,0,0]]).T
R_vacio=np.eye(3)

# Simbolicos
from sympy.matrices import Matrix
import sympy as sp
    # Generación de variables simbólicas
cos = sp.cos
sin = sp.sin
t, p, bb = sp.symbols("t p bb")
p1, p2, p3 = sp.symbols("p1 p2 p3")
q1, q2, q3, q4, q5, q6 = sp.symbols("q1 q2 q3 q4 q5 q6")
l1, l2, l3, l4, l5, l5 = sp.symbols("l1 l2 l3 l4 l5 l6")
d1, d2, d3, d4, d5, d5 = sp.symbols("d1 d2 d3 d4 d5 d6")

# ---------------------------------------------
# Pregunta 1
# Parte a)
A = np.array([[0, 0, 0, 0, 0, 1],
              [5**5, 5**4, 5**3, 5**2, 5, 1],
              [0, 0, 0, 0, 1, 0],
              [5*5**4, 4*5**3, 3*5**2, 2*5, 1, 0],
              [0, 0, 0, 2, 0, 0],
              [20*5**3, 12*5**2, 6*5, 2, 0, 0]])
b = np.array([0, 1, 0, 0, 0, 0])

x = np.linalg.inv(A).dot(b)
print(x)
a5=x[0]; a4=x[1]; a3=x[2]; a2=x[3]; a1=x[4]; a0=x[5]

# Verificación (no necesario)
t = np.arange(0, 5.01, 0.01)
s = a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0
ds = 5*a5*t**4 + 4*a4*t**3 + 3*a3*t**2 + 2*a2*t + a1
dds = 20*a5*t**3 + 12*a4*t**2 + 6*a3*t + 2*a2

plt.subplot(1,3,1); plt.plot(t,s); plt.grid()
plt.subplot(1,3,2); plt.plot(t,ds); plt.grid()
plt.subplot(1,3,3); plt.plot(t,dds); plt.grid()
plt.tight_layout()
plt.show()