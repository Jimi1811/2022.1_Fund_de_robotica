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

# ----------------------------------------------------
# Pregunta 1
# ----------------------------------------------------

# -- FUNCION - NO DESCOMENTAR
# Transformacion general
""" def S_T_dh_n(DH_tabla,n):
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
 """


# ------- a ---------
# 1. T simbolico
DH_tabla = sp.Matrix([
    [ 0,  q1,      1, -sp.pi/2],
    [q2,   0,      0,  sp.pi/2],
    [ 0,  q3,   1.41,        0]
])
S_Tf = S_T_dh_n(DH_tabla,3)

# print(S_Tf)

Matrix([[cos(q1 + q3), -sin(q1 + q3), 0, -1.0*q2*sin(q1) + 1.0*cos(q1) + 1.41*cos(q1 + q3)], 
        [sin(q1 + q3),  cos(q1 + q3), 0,  1.0*q2*cos(q1) + 1.0*sin(q1) + 1.41*sin(q1 + q3)], 
        [           0,             0, 1,                                                 0], 
        [           0,             0, 0,                                                 1]]) 

# 2. Evaluar en el q dado
Tf = S_Tf.subs({q1:1.57, q2:0.2, q3:1.57})
# Tf = np.array(Tf, dtype=np.float64)
# print (np.round(Tf,4))

# ------- c ---------
 # Se hara por newton

# 1. Funcion que me de el T
def fkine_P1(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    # Longitudes
    d1 = 0
    d2 = q2
    d3 = 0

    th1 = q1
    th2 = 0
    th3 = q3

    a1 = 1
    a2 = 0
    a3 = 1.41

    ap1 = -np.pi/2
    ap2 = np.pi/2
    ap3 = 0
    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = T_dh(d1, th1, a1, ap1)
    T2 = T_dh(d2, th2, a2, ap2)
    T3 = T_dh(d3, th3, a3, ap3)

    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3)
    return T

# 2. Funcion para obtener matriz jacobiana
def jacobian_P1(q, delta):
    # Crear una matriz 3x3
    J = np.zeros((2, 3))
    # Transformacion homogenea inicial (usando q)
    To = fkine_P1(q)
    To = To[0:2, -1:] # vector posicion

    # Iteracion para la derivada de cada columna
    for i in range(3):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i]+delta
        # Transformacion homogenea luego del incremento (q+delta)
        T = fkine_P1(dq)
        T = T[0:2, -1:] # vector posicion
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        Jq = 1/delta*(T-To)
        J[:, i:i+1] = Jq
    return J

# 3. Funcion para cinematica inversa
def ikine_P1(xdes, q0):
    # Error
    epsilon = 0.001
    # Maximas iteraciones
    max_iter = 100
    # Delta de la jacobiana
    delta = 0.00001
    # Copia de las articulaciones
    q = copy(q0)
    # Almacenamiento del error
    ee = []
    # Transformacion homogenea (usando q)
    To = fkine_P1(q)
    To = To[0:2, 3] # vector posicion

    for i in range(max_iter):
        # jacobiano
        J = jacobian_P1(q, delta)
        J = np.linalg.pinv(J)
        # Error entre el x deseado y x actual
        e = xdes - To
        # q_k+1
        q = q + np.dot(J,e)
        # Nueva mtransformada homogenea
        To = fkine_P1(q)
        To = To[0:2, 3] # vector posicion

        # Norma del error
        enorm = np.linalg.norm(e)
        ee.append(enorm)    # Almacena los errores
        # Condicion de termino
        if (enorm < epsilon):
            print("Error en la iteracion ",i, ": ", np.round(enorm,4))
            break
    return q

# Usando la funcion ikine_P1 para encontrar los valores de q
xdes = np.array([2,1])
q0 = np.array([0,0,0])
q_P1 = ikine_P1(xdes,q0)

print(q_P1)
print(fkine_P1(q_P1))

# ----------------------------------------------------
# Pregunta 2
# ----------------------------------------------------

 # ------- b ---------
res = 2*np.sqrt(2.3/5)
# print (res)
acel=0.5
# ------- c ---------
A = np.array([[0, 0, 0, 0, 0, 1],
              [6**5, 6**4, 6**3, 6**2, 6, 1],
              [0, 0, 0, 0, 1, 0],
              [5*6**4, 4*6**3, 3*6**2, 2*6, 1, 0],
              [0, 0, 0, 2, 0, 0],
              [20*6**3, 12*6**2, 6*6, 2, 0, 0]])
b = np.array([0.2, 2.5, 0, 0, 0, acel])

x = np.linalg.inv(A).dot(b)
print(x)
a5=x[0]; a4=x[1]; a3=x[2]; a2=x[3]; a1=x[4]; a0=x[5]

# Verificación (no necesario)
t = np.linspace(0, 6, 100)
s = a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0
ds = 5*a5*t**4 + 4*a4*t**3 + 3*a3*t**2 + 2*a2*t + a1
dds = 20*a5*t**3 + 12*a4*t**2 + 6*a3*t + 2*a2

plt.subplot(1,3,1); plt.plot(t,s); plt.grid()
plt.subplot(1,3,2); plt.plot(t,ds); plt.grid()
plt.subplot(1,3,3); plt.plot(t,dds); plt.grid()
plt.tight_layout()
plt.show() 

