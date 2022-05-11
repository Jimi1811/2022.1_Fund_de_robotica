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

########################################################
#                     PREGUNTA 2
########################################################

# T final de las 3 articulaciones por cinematica directa
DH_tabla = sp.Matrix([  
    [ 0,       q1, l1, -sp.pi/2],
    [q2, -sp.pi/2,  0, -sp.pi/2],
    [q3,  sp.pi/2,  0, -sp.pi/2]
])
S_Tf=S_T_dh_n(DH_tabla,3)
# print(S_Tf)

""" Matrix([[ sin(q1), -cos(q1),  0, l1*cos(q1) - q2*sin(q1) + q3*cos(q1)], 
            [-cos(q1), -sin(q1),  0, l1*sin(q1) + q2*cos(q1) + q3*sin(q1)], 
            [       0,        0, -1,                                    0], 
         [          0,        0,  0,                                    1]]) """

px, py = sp.symbols("px py")
# A*X = B
A = sp.Matrix([
    [-sin(q1),cos(q1)],
    [cos(q1), sin(q1)]
])

B = sp.Matrix([
    [px-l1*cos(q1)],
    [py-l1*sin(q1)]
])

print(sp.simplify((A.inv()*B)))
""" 
Matrix([[     -px*sin(q1) + py*cos(q1)], 
        [-l1 + px*cos(q1) + py*sin(q1)]]) """