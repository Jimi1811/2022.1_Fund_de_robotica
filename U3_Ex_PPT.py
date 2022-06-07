
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
l1, l2, l3, l4, l5, l5 = sp.symbols("l1 l2 l3 l4 l5 l6")
r, p, y, dr, dp, dy = sp.symbols("r p y dr dp dy")

# ---------------------------------
# Diapo 10
# ---------------------------------
'''
# Matriz de rotación
R = S_R_y(t)
# Derivada con respecto al ángulo t
dRdt = sp.diff(R,t)
# Matriz antisimétrica
w_hat = sp.simplify(dRdt*dt*R.T)
# print(w_hat)
# Vector de velocidad angular
w = sVectorFromSkew(w_hat)
print(w)
'''
# ---------------------------------
# Diapo 20 - Uso de rotaciones
# ---------------------------------
'''
# Todos respecto a base {0}

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
'''
# ---------------------------------
# Diapo 23 - derivadas analíticas
# ---------------------------------
'''
# Matriz de rotación de roll, pitch, yaw
R = sp.simplify(S_R_z(r)*S_R_y(p)*S_R_x(y))

# Derivada temporal de la matriz de rotación (usando la regla de la cadena)
dRdt = sp.simplify(sp.diff(R,r)*dr + sp.diff(R,p)*dp + sp.diff(R,y)*dy)

# Vel. angular antisimetrica
w_hat = sp.simplify(dRdt * R.T)
# Vel. Angular
w = sVectorFromSkew(w_hat)

print("w_hat: ",w_hat)
print("w: ", w)
'''
# ---------------------------------
# Diapo 32 - Jacobiano geometrico
# ---------------------------------
# Matrices de transformación homogénea
T01 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, l1*sp.cos(q1)],
                 [sp.sin(q1),  sp.cos(q1), 0, l1*sp.sin(q1)],
                 [         0,           0, 1,             0],
                 [         0,           0, 0,             1]])
T12 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, l2*sp.cos(q2)],
                 [sp.sin(q2),  sp.cos(q2), 0, l2*sp.sin(q2)],
                 [         0,           0, 1,             0],
                 [         0,           0, 0,             1]])
T02 = sp.simplify(T01*T12)

# Ejes z con respecto al sistema 0
z0 = sp.Matrix([[0],[0],[1]]); 
z1 = T01[0:3, 2]
# Puntos con respecto al sistema 0
p0 = sp.Matrix([[0],[0],[0]]); 
p1 = T01[0:3, 3]; 
p2 = T02[0:3, 3]; 

# Componentes de los Jacobianos
Jv1 = z0.cross(p2-p0); 
Jv2 = z1.cross(p2-p1); 
Jw1 = z0; 
Jw2 = z1

# Jacobiano geométrico (columna a columna)
J1 = sp.Matrix.vstack(Jv1, Jw1)
J2 = sp.Matrix.vstack(Jv2, Jw2)
# Jacobiano geométrico (completo)
J = sp.Matrix.hstack(J1, J2)

print("J:",J)
print("Rango:", sp.Matrix.rank(J))
