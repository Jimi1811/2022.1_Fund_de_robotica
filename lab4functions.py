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

# - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 

def dh(d, theta, a, alpha):
    c_th = cos(theta)
    s_th = sin(theta)
    c_ap = cos(alpha)
    s_ap = sin(alpha)

    T = np.array([[c_th, -c_ap*s_th,  s_ap*s_th, a*c_th],
                    [s_th,  c_ap*c_th, -s_ap*c_th, a*s_th],
                    [0,     s_ap,       c_ap,      d],
                    [0,     0,          0,         1]])
    return T


def fkine_ur5(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]

    # Longitudes (en metros)
    d1 = 0.0892
    d2 = 0
    d3 = 0
    d4 = 0.1093
    d5 = 0.09475
    d6 = 0.0825

    th1 = 0 + q1
    th2 = q2 + pi
    th3 = q3
    th4 = q4 + pi
    th5 = pi + q5
    th6 = q6

    a1 = 0
    a2 = 0.425
    a3 = 0.392
    a4 = 0
    a5 = 0
    a6 = 0

    ap1 = pi/2
    ap2 = 0
    ap3 = 0
    ap4 = pi/2
    ap5 = pi/2
    ap6 = 0

    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = dh(d1, th1, a1, ap1)
    T2 = dh(d2, th2, a2, ap2)
    T3 = dh(d3, th3, a3, ap3)
    T4 = dh(d4, th4, a4, ap4)
    T5 = dh(d5, th5, a5, ap5)
    T6 = dh(d6, th6, a6, ap6)

    print('T1')
    print(np.round(T1, 3))
    print('T2')
    print(np.round(T2, 3))
    print('T3')
    print(np.round(T3, 3))
    print('T4')
    print(np.round(T4, 3))
    print('T5')
    print(np.round(T5, 3))
    print('T6')
    print(np.round(T6, 3))

    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
    return T

def jacobian_ur5(q, delta):
    # delta=0.0001

    # Crear una matriz 3x6
    J = np.zeros((3, 6))
    # Transformacion homogenea inicial (usando q)
    To = fkine_ur5(q)
    To = To[0:3, -1:] # vector posicion

    # Iteracion para la derivada de cada columna
    for i in range(6):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i]+delta
        # Transformacion homogenea luego del incremento (q+delta)
        T = fkine_ur5(dq)
        T = T[0:3, -1:] # vector posicion
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        Jq = 1/delta*(T-To)
        J[:, i:i+1] = Jq
    return J

def ikine_ur5(xdes, q0):
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
    To = fkine_ur5(q)
    To = To[0:3, 3] # vector posicion
    # Resetear cuando se llega a la cantidad maxima de iteraciones
    restart = True

    while restart:
        for i in range(max_iter):
            # Hacer el for 1 vez
            restart = False
            # Pseudo-inversa del jacobiano
            J = jacobian_ur5(q, delta)
            J = np.linalg.pinv(J)
            # Error entre el x deseado y x actual
            e = xdes - To
            # q_k+1
            q = q + np.dot(J,e)
            # Nueva mtransformada homogenea
            To = fkine_ur5(q)
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