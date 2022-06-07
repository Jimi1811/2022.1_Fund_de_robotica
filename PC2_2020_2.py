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

# ------------------------------------------------------
# PREGUNTA 1
# -----------------------------------------------------

# -------  a  ---------
 
# 1. T simbolico
DH_tabla = sp.Matrix([
    [ q1,  0,   0,        0],
    [  0, q2,   0, -sp.pi/2],
    [0.6, q3,   0,  sp.pi/2],
    [  0, q4, 0.8,       0]
])
S_Tf = S_T_dh_n(DH_tabla,4)

# print(S_Tf)
""" Matrix([[-sin(q2)*sin(q4) + cos(q2)*cos(q3)*cos(q4), -sin(q2)*cos(q4) - sin(q4)*cos(q2)*cos(q3), sin(q3)*cos(q2), -0.8*sin(q2)*sin(q4) - 0.6*sin(q2) + 0.8*cos(q2)*cos(q3)*cos(q4)], 
        [ sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2), -sin(q2)*sin(q4)*cos(q3) + cos(q2)*cos(q4), sin(q2)*sin(q3), 0.8*sin(q2)*cos(q3)*cos(q4) + 0.8*sin(q4)*cos(q2) + 0.6*cos(q2)], 
        [                          -sin(q3)*cos(q4),                            sin(q3)*sin(q4),         cos(q3),                                        q1 - 0.8*sin(q3)*cos(q4)], 
        [                                         0,                                          0,               0,                                                               1]])
 """

# 2. Evaluar en el q dado
Tf = S_Tf.subs({q1:1, q2:0., q3:-sp.pi/2, q4:-sp.pi/2})

# print (Tf)
""" Matrix([[ 0, 0, -1,    0], 
        [-1, 0,  0, -0.2], 
        [ 0, 1,  0,    1], 
        [ 0, 0,  0,    1]])
 """

# -------  c  ---------
# Se hara por newton

# 1. Funcion que me de el T
def fkine_P1(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]

    # Longitudes
    d1 = q1
    d2 = 0
    d3 = 0.6
    d4 = 0

    th1 = 0
    th2 = q2
    th3 = q3
    th4 = q4

    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0.8

    ap1 = 0
    ap2 = -np.pi/2
    ap3 = np.pi/2
    ap4 = 0
    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = T_dh(d1, th1, a1, ap1)
    T2 = T_dh(d2, th2, a2, ap2)
    T3 = T_dh(d3, th3, a3, ap3)
    T4 = T_dh(d4, th4, a4, ap4)

    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4)
    return T

# 2. Funcion para obtener matriz jacobiana
def jacobian_P1(q, delta):
    # Crear una matriz 3x4
    J = np.zeros((3, 4))
    # Transformacion homogenea inicial (usando q)
    To = fkine_P1(q)
    To = To[0:3, -1:] # vector posicion

    # Iteracion para la derivada de cada columna
    for i in range(4):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i]+delta
        # Transformacion homogenea luego del incremento (q+delta)
        T = fkine_P1(dq)
        T = T[0:3, -1:] # vector posicion
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        Jq = 1/delta*(T-To)
        J[:, i:i+1] = Jq
    return J

# 3. Funcion para cinematica inversa
def ikine_P1(xdes, q0):
    # Error
    epsilon = 0.001
    # Maximas iteraciones
    max_iter = 1000
    # Delta de la jacobiana
    delta = 0.00001
    # Copia de las articulaciones
    q = copy(q0)
    # Almacenamiento del error
    ee = []
    # Transformacion homogenea (usando q)
    To = fkine_P1(q)
    To = To[0:3, 3] # vector posicion
    # Resetear cuando se llega a la cantidad maxima de iteraciones
    restart = True

    while restart:
        for i in range(max_iter):
            # Hacer el for 1 vez
            restart = False
            # Pseudo-inversa del jacobiano
            J = jacobian_P1(q, delta)
            J = np.linalg.pinv(J)
            # Error entre el x deseado y x actual
            e = xdes - To
            # q_k+1
            q = q + np.dot(J,e)
            # Nueva mtransformada homogenea
            To = fkine_P1(q)
            To = To[0:3, 3] # vector posicion

            # Norma del error
            enorm = np.linalg.norm(e)
            ee.append(enorm)    # Almacena los errores
            # Condicion de termino
            if (enorm < epsilon):
                print("Error en la iteracion ",i, ": ", np.round(enorm,4))
                break
            if (i==max_iter-1 and enorm > epsilon):
                print("Iteracion se repite")
                print("Error en la iteracion ",i, ": ", enorm)
                restart = True
    return q

# Usando la funcion ikine_P1 para encontrar los valores de q
xdes = np.array([-0.0343,0.4,0.2])
q0 = np.array([0.4, 1, 1, 1])
q_P1 = ikine_P1(xdes,q0)

print(q_P1)