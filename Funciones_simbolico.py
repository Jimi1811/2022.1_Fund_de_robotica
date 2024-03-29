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
def S_T_dh(d, th, a, alpha):
    cth = sp.cos(th) 
    sth = sp.sin(th)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)
    Tdh = sp.Matrix([[cth, -ca*sth,  sa*sth, a*cth],
                     [sth,  ca*cth, -sa*cth, a*sth],
                     [0,        sa,     ca,      d],
                     [0,         0,      0,      1]])
    return Tdh
    
######################################################
#     CINEMATICA DIFERENCIAL DE ROBOTS MANIPULADORES
######################################################

# Velocidad angular extraido de velocidad angular antisimetrica
def S_w_de_w_anti(S):
    return sp.Matrix([S[2,1],S[0,2],S[1,0]])

# Transformacion general
def S_T_dh_n(DH_tabla,n):
    DH = sp.zeros(6,4)
    DH[:n,:] = DH_tabla[:,:]

    Tf = sp.eye(4)

    for i in range(n):
        cth = sp.cos(DH[i,1])
        sth = sp.sin(DH[i,1])
        ca = sp.cos(DH[i,3])
        sa = sp.sin(DH[i,3])

        a = DH[i,2]
        d = DH[i,0]

        Ti = np.array([[cth, -ca*sth,  sa*sth, a*cth], 
                    [sth,  ca*cth, -sa*cth, a*sth], 
                    [0,     sa,       ca,      d], 
                    [0,     0,          0,         1]]) 
        Tf = sp.simplify(Tf*Ti)
    Tf = sp.simplify(Tf)

    return Tf

######################################################
#     CINEMATICA INVERSA DE ROBOTS MANIPULADORES
######################################################

# ---------------------------------------------------
#               Simbolico de T deseado
# ---------------------------------------------------  
nx, ny, nz, ox, oy, oz, ax, ay, az, px, py, pz= sp.symbols("nx ny nz ox oy oz ax ay az px py pz")

def S_T_des():
    Tdes = sp.Matrix([[nx, ox, ax, px],
                      [ny, oy, ay, py],
                      [nz, oz, az, pz],
                      [0,   0,  0,  1]])
    return Tdes


######################################################
#            CINEMATICA DE ROBOTS MOVILES
######################################################

# --------------------------------
# Rueda Omnidireccional (Sweedish)

#  Restriccion de rodadura
def S_Omni_RR(alpha,beta,gamma,l):
    A = sp.Matrix([[sp.sin(alpha+beta+gamma), -sp.cos(alpha+beta+gamma), -l*sp.cos(beta+gamma)]])
    return A 