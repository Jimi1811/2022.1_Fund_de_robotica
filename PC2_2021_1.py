from Funciones import *  
from Funciones_simbolico import *  

# Para limpiar terminal
import os 
os.system("clear") 

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

# ---------------------------------------------------------

# PREGUNTA 2

# Usando cinematica directa, se halla la transformada de cada artuculacion
T1 = S_T_dh(l1,q1,0,sp.pi/2)
T2 = S_T_dh(0,q2,0,-sp.pi/2)
T3 = S_T_dh(q3,0,0,0)

# T total
Tf = sp.simplify(T1*T2*T3)

print(Tf)
""" Matrix([[cos(q1)*cos(q2), -sin(q1), -sin(q2)*cos(q1), -q3*sin(q2)*cos(q1)],
            [sin(q1)*cos(q2),  cos(q1), -sin(q1)*sin(q2), -q3*sin(q1)*sin(q2)], 
            [sin(q2),                0,          cos(q2),     l1 + q3*cos(q2)], 
            [0,                      0,                0,                   1]])
"""

Sol_Tf = Tf.subs([ (q1,0), (q2,0), (q3,0), (l1,40)])