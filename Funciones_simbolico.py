######################################################
#       PROPIEDADES TRIGONOMETRICAS
######################################################
"""
sen(a+b) = sen(a) cos(b) + sen(b) cos(a)
sen(a-b) = sen(a) cos(b) - sen(b) cos(a)

cos(a+b) = cos(a) cos(b) - sen(a) sen(b)
cos(a-b) = cos(a) cos(b) + sen(a) sen(b)
"""

# from Funciones import *  DESCOMENTAR
# from Funciones_simbolico import *  DESCOMENTAR

# Para limpiar terminal
import os 
os.system("clear") 

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


######################################################
#                  PARA SIMBOLOS
######################################################

# ---------------------------------------------------
#              Matriz de rotacion R
# ---------------------------------------------------
# R_x
def S_R_x(ang):
    Rx = sp.Matrix([
        [1, 0, 0],
        [0, cos(ang), -sin(ang)],
        [0, sin(ang), cos(ang)]
        ])
    return Rx
# R_y
def S_R_y(ang):
    Ry = sp.Matrix([[cos(ang), 0, sin(ang)],
                    [0, 1, 0],
                    [-sin(ang), 0, cos(ang)]])
    return Ry
# R_z
def S_R_z(ang):
    Rz = sp.Matrix([[cos(ang), -sin(ang), 0],
                   [sin(ang), cos(ang), 0],
                   [0,0,1]])
    return Rz

# ---------------------------------------------------
#          Transformacion homogenea T
# ---------------------------------------------------

# Traslacion pura
def S_T_tra(x, y, z):
    T = sp.Matrix([[1,0,0,x],
                   [0,1,0,y],
                   [0,0,1,z],
                   [0,0,0,1]])
    return T

# Rotacion pura
def S_T_rot_x(ang):
    T = sp.Matrix([[1, 0,0,0],
                   [0, sp.cos(ang),-sp.sin(ang),0],
                   [0, sp.sin(ang), sp.cos(ang),0],
                   [0, 0, 0, 1]])
    return T

def S_T_rot_y(ang):
    T = sp.Matriz([[sp.cos(ang), 0, sp.sin(ang), 0],
                  [0,            1,           0, 0],
                  [-sp.sin(ang), 0, sp.cos(ang), 0],
                  [0,            0,           0, 1]])
    return T

def S_T_rot_z(ang):
    T = sp.Matrix([[sp.cos(ang),-sp.sin(ang),0,0],
                   [sp.sin(ang), sp.cos(ang),0,0],
                   [0,                     0,1,0],
                   [0,                     0,0,1]])
    return T

######################################################
#     CINEMATICA DIRECTA DE ROBOTS MANIPULADORES
######################################################

# ---------------------------------------------------
#                 Denavit-Hartenberg
# ---------------------------------------------------  

# Transformacion general
def S_T_DH(d, th, a, alpha):
    cth = sp.cos(th) 
    sth = sp.sin(th)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)
    Tdh = sp.Matrix([[cth, -ca*sth,  sa*sth, a*cth],
                     [sth,  ca*cth, -sa*cth, a*sth],
                     [0,        sa,     ca,      d],
                     [0,         0,      0,      1]])
    return Tdh