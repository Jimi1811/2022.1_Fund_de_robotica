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
q1, q2, q3, q4, q5, q6, q7 = sp.symbols("q1 q2 q3 q4 q5 q6 q7")
l1, l2, l3, l4, l5, l6, L = sp.symbols("l1 l2 l3 l4 l5 l6 L")
r, p, y, dr, dp, dy = sp.symbols("r p y dr dp dy")
d1, d2, d3, d4, d5, d5 = sp.symbols("d1 d2 d3 d4 d5 d6")
vx, vy, vz, wx, wy, wz, v, w = sp.symbols("vx vy vz wx wy wz v w")

#############################################################
# PREGUNTA 1
#############################################################

# ------------------
# Consigna a

# Distancias
d1 = 0.105
d3 = 0.25
d5 = 0.25
d7 = 0.103
# Matrices de transformación homogénea i con respecto a i-1
T01 = S_T_dh(d1,q1, 0,  sp.pi/2)
T12 = S_T_dh(q2,0,  0, -sp.pi/2)
T23 = S_T_dh(d3,q3, 0,  sp.pi/2)
T34 = S_T_dh(q4,0,  0,  sp.pi/2)
T45 = S_T_dh(d5,q5, 0, -sp.pi/2)
T56 = S_T_dh(0 ,q6, 0,  sp.pi/2)
T67 = S_T_dh(d7,q7, 0,        0)

# Matrices de transformación homogénea con respecto a 0
T02 = sp.simplify(T01*T12)
T03 = sp.simplify(T02*T23)
T04 = sp.simplify(T03*T34)
T05 = sp.simplify(T04*T45)
T06 = sp.simplify(T05*T56)
T07 = sp.simplify(T06*T67)

# Ejes z (con respecto al sistema 0)
z0 = sp.Matrix([[0],[0],[1]]);
z1 = T01[0:3, 2]
z2 = T02[0:3, 2]
z3 = T03[0:3, 2]
z4 = T04[0:3, 2]
z5 = T05[0:3, 2]
z6 = T06[0:3, 2]

# Puntos con respecto al sistema 0
p0 = sp.Matrix([0,0,0])
p1 = T01[0:3, 3]
p2 = T02[0:3, 3]
p3 = T03[0:3, 3]
p4 = T04[0:3, 3]
p5 = T05[0:3, 3]
p6 = T06[0:3, 3]
p7 = T07[0:3, 3]

# Componentes del Jacobiano (velocidad lineal)
Jv1 = sp.simplify(z0.cross(p7-p0))
Jv2 = sp.simplify(z1)
Jv3 = sp.simplify(z2.cross(p7-p2))
Jv4 = sp.simplify(z3)
Jv5 = sp.simplify(z4.cross(p7-p4))
Jv6 = sp.simplify(z5.cross(p7-p5))
Jv7 = sp.simplify(z6.cross(p7-p6))
# Componentes del Jacobiano (velocidad angular)
Jw1 = z0 
Jw2 = sp.Matrix([[0],[0],[0]]) 
Jw3 = z2 
Jw4 = sp.Matrix([[0],[0],[0]])
Jw5 = z4 
Jw6 = z5 
Jw7 = z6
# Jacobiano geométrico
J1 = sp.Matrix.vstack(Jv1, Jw1)
J2 = sp.Matrix.vstack(Jv2, Jw2)
J3 = sp.Matrix.vstack(Jv3, Jw3)
J4 = sp.Matrix.vstack(Jv4, Jw4)
J5 = sp.Matrix.vstack(Jv5, Jw5)
J6 = sp.Matrix.vstack(Jv6, Jw6)
J7 = sp.Matrix.vstack(Jv7, Jw7)
J = sp.Matrix.hstack(J1, J2, J3, J4, J5, J6, J7)

# Jacobiano en el punto dado:
# Js = J.subs({q1:0, q2:0, q3:0, q4:0, q5:0, q6:0, q7:0})

# print("J(q):\n", Js) 
# >>
# Matrix([[0,  0, 0,  0,  0, 0.103,  0], 
#         [0, -1, 0, -1,  0,     0,  0], 
#         [0,  0, 0,  0,  0,     0,  0], 
#         [0,  0, 0,  0,  0,     0,  0], 
#         [0,  0, 0,  0,  0,    -1,  0], 
#         [1,  0, 1,  0, -1,     0, -1]
#         ])

# ---------------------
#  consigna b

# print(sp.shape(Js))
""" Js1 = sp.Matrix([[0,  0, 0,  0,  0, 0.103,  0], 
        [0, -1, 0, -1,  0,     0,  0], 
        [0,  0, 0,  0,  0,     0,  0], 
        [0,  0, 0,  0,  0,     0,  0], 
        [0,  0, 0,  0,  0,    -1,  0], 
        [1,  0, 1,  0, -1,     0, -1]
        ])
# Js1 = Js1[:,:6]
# print("rango: ",Js1.rank())
# print(sp.shape(Js1))

# -----------------------
# Consigna c

# n = gdl = 7 
# m = eq = 6
V = sp.Matrix([0.5, 0.5, 0.3, 0, 0, 0])

# Pseudo inversa
Jpp = (Js1*(Js1.T)).inv()
Jp = sp.simplify(Js1.T*Jpp)
# Velocidades articulares
dq = Jp*V """

# print(dq)

# -----------------------
# Consigna d

Jf = J.subs({q1:0, q2:0.5, q3:0, q4:0, q5:0.1, q6:0.1, q7:0.1})
F = sp.Matrix([1, 1, 0, 0, 0, 0])

t = Jf.T*F
print(t)
# >>
Matrix([[0.511258041777122], 
        [-1], 
        [0.0112580417771217], 
        [-1], 
        [-0.0112580417771217], 
        [0.0917419582228783], 
        [0]
        ])