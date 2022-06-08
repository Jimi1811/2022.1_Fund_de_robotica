######################################################
#       PROPIEDADES TRIGONOMETRICAS
######################################################
"""
sen(a+b) = sen(a) cos(b) + sen(b) cos(a)
sen(a-b) = sen(a) cos(b) - sen(b) cos(a)

cos(a+b) = cos(a) cos(b) - sen(a) sen(b)
cos(a-b) = cos(a) cos(b) + sen(a) sen(b)
"""

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
th, t, p, bb, dt = sp.symbols("th t p bb dt")
p1, p2, p3, p4, p5, p6 = sp.symbols("p1 p2 p3 p4 p5 p6")
q1, q2, q3, q4, q5, q6 = sp.symbols("q1 q2 q3 q4 q5 q6")
l1, l2, l3, l4, l5, l6, L = sp.symbols("l1 l2 l3 l4 l5 l6 L")
r, p, y, dr, dp, dy = sp.symbols("r p y dr dp dy")
d1, d2, d3, d4, d5, d5 = sp.symbols("d1 d2 d3 d4 d5 d6")
vx, vy, vz, wx, wy, wz, v, w = sp.symbols("vx vy vz wx wy wz v w")


# ---------------------------------------------
""" # Pregunta 1
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
plt.show() """

# ------------------------------------------------
# Pregunta 2 
# ------------------------------------------------

# parte a)
L = sp.sqrt(0.18**2+0.14**2)
alfa1 = sp.atan(0.18/0.14)

# Para cada rueda
L1 = L; a1 = -alfa1; b1 = sp.pi/2+alfa1; g1 = sp.pi/4
L2 = L; a2 = alfa1; b2 = sp.pi/2-alfa1; g2 = -sp.pi/4
L3 = L; a3 = sp.pi-alfa1; b3 = -(sp.pi/2-alfa1); g3 = sp.pi/4
L4 = L; a4 = sp.pi+alfa1; b4 = -(sp.pi/2+alfa1); g4 = -sp.pi/4

A = sp.Matrix.vstack(S_Omni_RR(a1,b1,g1,L),
                     S_Omni_RR(a2,b2,g2,L),
                     S_Omni_RR(a3,b3,g3,L),
                     S_Omni_RR(a4,b4,g4,L))
A = sp.simplify(A)
# print(sp.shape(A))

# Para B
r = 0.06
# B = sp.Matrix([[r*sp.cos(np.pi/4), 0, 0, 0],
#                [0, r*sp.cos(np.pi/4), 0, 0],
#                [0, 0, r*sp.cos(np.pi/4), 0],
#                [0, 0, 0, r*sp.cos(np.pi/4)]])

B = r*sp.cos(np.pi/4)*sp.eye(4)
B = sp.simplify(B)

# print("Matriz A: \n", A)
# print("\nMatriz B: \n", B)

Xi = sp.Matrix([[vx],[vy],[w]])
Phi = sp.simplify( (B.T*B).inv()*B.T*A*Xi )
Phi = Phi.subs({'vx': 0.2, 'vy': 0.346, 'w': 0})

# print("\nVelocidades de giro: \n", Phi)

# Parte b)

p1, p2, p3, p4 = sp.symbols("p1 p2 p3 p4")
Phi = sp.Matrix([[p1],[p2],[p3],[p4]])
Xi = sp.simplify((A.T*A).inv()*A.T*B*Phi)

Xi1 = sp.Matrix([[0.0106066017177982*sp.sqrt(2)*(p1 + p2 + p3 + p4)], 
        [0.0106066017177982*sp.sqrt(2)*(p1 + p2 + p3 + p4)], 
        [0.0331456303681194*sp.sqrt(2)*(p1 + p2 + p3 + p4)]])

Xi1 = Xi1.subs({'p1': 20, 'p2': 20, 'p3': 20, 'p4': 20})
print(Xi1)