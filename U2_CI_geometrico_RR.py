from Funciones import *  
from Funciones_simbolico import *  

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

# ---------------------------------------------------------------------

#class RobotRR(object):
def cinematica_directa_RR(q1, q2, L1, L2):
    x = L1*np.cos(q1) + L2*np.cos(q1+q2)
    y = L1*np.sin(q1) + L2*np.sin(q1+q2)
    return (x,y)

# Función que realiza el cálculo de la cinemática inversa (calculado usando geometría)
def cinematica_inversa_RR_geom(x, y, L1, L2):
    c2 = (x**2+y**2-L1**2-L2**2)/(2*L1*L2)
    s2a =  np.sqrt(1-c2**2)
    s2b = -np.sqrt(1-c2**2)
    # Solución 1:
    q2a = np.arctan2(s2a, c2)
    q1a = np.arctan2(y,x) - np.arctan2(L2*s2a, L1+L2*c2)
    # Solución 2:
    q2b = np.arctan2(s2b, c2)
    q1b = np.arctan2(y,x) - np.arctan2(L2*s2b, L1+L2*c2)
    # Retornar ambas soluciones
    return ((q1a, q2a), (q1b, q2b))
    

# Prueba de la cinemática inversa
# -------------------------------

L1 = 1.0; L2 = 1.0  # Longitudes fijas
xdes = 1.2          # Valor deseado en x
ydes = 1.2          # Valor deseado en y

# Cálculo de la cinemática inversa
Q = cinematica_inversa_RR_geom(xdes, ydes, L1, L2)

print("Solución 1:", np.round(Q[0], 4))
print("Solución 2:", np.round(Q[1], 4))