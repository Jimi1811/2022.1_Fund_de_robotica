
######################################################
#           LIBRERIAS - siempre pegar
######################################################
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
th, t, p, bb, dt = sp.symbols("th t p bb dt")
p1, p2, p3 = sp.symbols("p1 p2 p3")
q1, q2, q3, q4, q5, q6 = sp.symbols("q1 q2 q3 q4 q5 q6")
l1, l2, l3, l4, l5, l6, L = sp.symbols("l1 l2 l3 l4 l5 l6 L")
r, p, y, dr, dp, dy = sp.symbols("r p y dr dp dy")

# ---------------------------------
# Diapo 10
# --------------------------------
""" 
# Matriz de rotación
R = S_R_y(t)
# Derivada con respecto al ángulo t
dRdt = sp.diff(R,t)
# Matriz antisimétrica
w_hat = sp.simplify(dRdt*dt*R.T)
# print(w_hat)
# Vector de velocidad angular
w = S_w_de_w_antiw_hat)
print(w)

 """
 # ---------------------------------
# Diapo 20 - Uso de rotaciones
# ---------------------------------

# Todos respecto a base {0}
""" 
# Componente de velocidad debido a "roll"
wr = sp.Matrix([0,0,1])*dr

# Componente de velocidad debido a "pitch"
Rz = S_R_z(r)
wp = Rz[:,1]*dp

# Componente de velocidad debido a "yaw"
Rrp = sp.simplify(S_R_z(r)*S_R_y(p))
wy = Rrp[:,0]*dy

# Velocidad angular
w = wr + wp + wy
print(w)

# Matriz E de w = Eo * x'

# Es manual pero puedo sacarlo dea la misma matriz
 """
# ---------------------------------
# Diapo 23 - derivadas analíticas
# ---------------------------------

""" # Matriz de rotación de roll, pitch, yaw
R = sp.simplify(S_R_z(r)*S_R_y(p)*S_R_x(y))

# Derivada temporal de la matriz de rotación (usando la regla de la cadena)
dRdt = sp.simplify(sp.diff(R,r)*dr + sp.diff(R,p)*dp + sp.diff(R,y)*dy)

# Vel. angular antisimetrica
w_hat = sp.simplify(dRdt * R.T)
# Vel. Angular
w = S_w_de_w_anti(w_hat)

print("w_hat: \n",w_hat,"\n")
print("w: \n", w,"\n")
 """
# ---------------------------------
# Diapo 32 - Jacobiano geometrico
# ---------------------------------
""" 
# Matrices de transformación homogénea

# 1. T de 0 a 1 
T01 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, l1*sp.cos(q1)],
                 [sp.sin(q1),  sp.cos(q1), 0, l1*sp.sin(q1)],
                 [         0,           0, 1,             0],
                 [         0,           0, 0,             1]])
T12 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, l2*sp.cos(q2)],
                 [sp.sin(q2),  sp.cos(q2), 0, l2*sp.sin(q2)],
                 [         0,           0, 1,             0],
                 [         0,           0, 0,             1]])
# 2. T de 0 a 2
T02 = sp.simplify(T01*T12)

# Ejes z con respecto al sistema 0
z0 = sp.Matrix([[0],[0],[1]]); 
z1 = T01[0:3, 2]
# Puntos con respecto al sistema 0
p0 = sp.Matrix([[0],[0],[0]]); 
p0_1 = T01[0:3, 3]; 
p2 = T02[0:3, 3]; 
p0_2 = p2-p0

p1_2 = p0_2 - p0_1
# Componentes de los Jacobianos
Jv1 = z0.cross(p0_2); 
Jv2 = z1.cross(p1_2); 
Jw1 = z0; 
Jw2 = z1

# Jacobiano geométrico (columna a columna)
J1 = sp.Matrix.vstack(Jv1, Jw1)
J2 = sp.Matrix.vstack(Jv2, Jw2)
# Jacobiano geométrico (completo)
J = sp.Matrix.hstack(J1, J2)

print("J:",J)
print("Rango:", sp.Matrix.rank(J)) """

# ---------------------------------
# Diapo 33 - 34 : Jacobiano geometrico - DH - 4 varbiales
# ---------------------------------

""" # Matrices de transformación homogénea i con respecto a i-1
T01 = S_T_dh(0, q1, 0, sp.pi/2)
T12 = S_T_dh(0, q2, 0, sp.pi/2)
T23 = S_T_dh(L, q3, 0, sp.pi/2)
T34 = S_T_dh(0, q4, L, 0)

# Matrices de transformación homogénea con respecto a 0
T02 = sp.simplify(T01*T12)
T03 = sp.simplify(T02*T23)
T04 = sp.simplify(T03*T34)

# Ejes z (con respecto al sistema 0)
z0 = sp.Matrix([[0],[0],[1]]);
z1 = T01[0:3, 2]
z2 = T02[0:3, 2]
z3 = T03[0:3, 2]

# Puntos con respecto al sistema 0
p0 = sp.Matrix([0,0,0])
p1 = T01[0:3, 3]
p2 = T02[0:3, 3]
p3 = T03[0:3, 3]
p4 = T04[0:3, 3]


# Componentes del Jacobiano (velocidad lineal)
Jv1 = sp.simplify(z0.cross(p4-p0))
Jv2 = sp.simplify(z1.cross(p4-p1))
Jv3 = sp.simplify(z2.cross(p4-p2))
Jv4 = sp.simplify(z3.cross(p4-p3))
# Componentes del Jacobiano (velocidad angular)
Jw1 = z0; Jw2 = z1; Jw3 = z2; Jw4 = z3

# Jacobiano geométrico
J1 = sp.Matrix.vstack(Jv1, Jw1)
J2 = sp.Matrix.vstack(Jv2, Jw2)
J3 = sp.Matrix.vstack(Jv3, Jw3)
J4 = sp.Matrix.vstack(Jv4, Jw4)
J = sp.Matrix.hstack(J1, J2, J3, J4)

print("J1:", J1)
print("J2:", J2)
print("J3:", J3)
print("J4:", J4)

# Jacobiano en el punto dado:
Js = J.subs({q1:0, q2:135*sp.pi/180, q3:sp.pi, q4:sp.pi})

print("J(q):", Js) """

# ---------------------------------
# Diapo 40 - Jacobiano analitico
# ---------------------------------
""" 
# Forma 1
# =======

# Cinemática directa (término a término)
x = l1*sp.cos(q1) + l2*sp.cos(q1+q2)
y = l1*sp.sin(q1) + l2*sp.sin(q1+q2)
phi = q1 + q2

# Derivadas (término a término)
dxdq1 = sp.diff(x, q1) 
dxdq2 = sp.diff(x, q2)
dydq1 = sp.diff(y, q1) 
dydq2 = sp.diff(y, q2)
dphidq1 = sp.diff(phi, q1) 
dphidq2 = sp.diff(phi, q2)

# Jacobiano analitico
Ja1 = sp.Matrix([[  dxdq1,   dxdq2],
                [  dydq1,   dydq2],
                [dphidq1, dphidq2]])
#print(Ja1)

# Forma 2
# =======

# Cinemática directa (como vector)
X = sp. Matrix([[ l1*sp.cos(q1) + l2*sp.cos(q1+q2)],
                [ l1*sp.sin(q1) + l2*sp.sin(q1+q2)],
                [ q1 + q2]])

# Vector de variables articulares
q = sp.Matrix([q1,q2])

# Jacobiano analítico (usando la función "jacobian")
Ja1 = X.jacobian(q)
print(Ja1) 


############ SINGULARIDAD #############
# Extraer solo la parte correspondiente a la posición
Jaxy = Ja1[0:2,0:2]
# Determinante
det = sp.simplify(Jaxy.det())
# Rango
r = Jaxy.rank()

print("El determinante es:", det)
print("El rango es:", r)


# Caso singular 1: reemplazando q2=0
J1 = sp.simplify(Jaxy.subs({q2: 0}))
# Determinante cuando q2=0:
det1 = J1.det()
# Rango cuando q2=0
r1 = J1.rank()

print("\n Jacobiano analítico cuando q2=0: \n ", J1)
print("Determinante cuando q2=0:", det1)
print("Rango cuando q2=0:", r1)

# Caso singular 2: reemplazando q2=pi
J2 = sp.simplify(Jaxy.subs({q2:sp.pi}))
# Determinante cuando q2=0:
det2 = J2.det()
# Rango cuando q2=pi
r2 = J2.rank()

print("\n Jacobiano analítico cuando q2=pi: \n ", J2)
print("Determinante cuando q2=pi:", det2)
print("Rango cuando q2=pi:", r2) """


# ---------------------------------
# Diapo 42 - Jacobiano analitico
# ---------------------------------

# Cinemática directa
X = sp.Matrix([[q3*sp.cos(q2)*sp.cos(q1)],
               [q3*sp.cos(q2)*sp.sin(q1)],
               [d1 + q3*sp.sin(q2)]])

# Variables articulares
q = sp.Matrix([q1, q2, q3])

# Jacobiano analítico
Ja2 = X.jacobian(q)
print(Ja2)


############ SINGULARIDAD #############
# Determinante
det = sp.simplify(Ja2.det())
# Rango
r = Ja2.rank()

print("El determinante es:",det)
print("El rango es:", r)

# Singularidad 1: cuando q2=90
J1 = sp.simplify(Ja2.subs({q2:sp.pi/2}))
# Determinante
det1 = J1.det()
# Rango
r1 = J1.rank()

print("Jacobiano analítico cuando q2=pi:\n", J1)
print("Determinante:",det1)
print("Rango:", r1)

# Singularidad 2: cuando q2=-90
J2 = sp.simplify(Ja2.subs({q2:-sp.pi/2}))
# Determinante
det2 = J2.det()
# Rango
r2 = J2.rank()

print("Jacobiano analítico cuando q2=-pi:\n",J2)
print("Determinante:",det2)
print("Rango:", r2)
# Singularidad 3: cuando q3=0
J3 = sp.simplify(Ja2.subs({q3:0}))
# Determinante
det3 = J3.det()
# Rango
r3 = J3.rank()

print("Jacobiano analítico cuando q3=0:\n",J3)
print("Determinante:",det3)
print("Rango:", r3)

