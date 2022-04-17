from Funciones import *

# Para limpiar terminal
import os 
os.system("clear") 

# Usar simbolicos
from sympy.matrices import Matrix
import sympy as sp

# Realizar operaciones
import numpy as np

#Generación de variables simbólicas
cos = sp.cos
sin = sp.sin
t, p, bb = sp.symbols("t p bb")
p1, p2, p3 = sp.symbols("p1 p2 p3")

# Generalizar transf. homogenea T
d_vacio=np.array([[0,0,0]]).T
R_vacio=np.eye(3)

############################################################

# Pregunta 1

R_i_o = np.array([
    [0 , 0.5 , 0.866],
    [1 , 0 , 0],
    [0 , 0.866 , -0.5]
    ])

R_i_f = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,-1]
])

R_o_f = R_i_o.T.dot(R_i_f)
# print(np.round(R_o_f,2))

u1,th1 = eje_angulo_pos(R_o_f)
# print("vector u: ", u1)
# print("Angulo en deg: ", np.round(th1/np.pi*180,2))
u2,th2 = eje_angulo_neg(R_o_f)
# print("vector u: ", u2)
# print("Angulo en deg: ", np.round(th2/np.pi*180,2))

T_i_f = T(R_i_f,np.array([[2,4,-3]]).T)
# print(np.round(T_i_f,2))

T_o_i = np.linalg.inv(T(R_i_o,d_vacio))
# print(np.round(T_o_i,2))

T_o_f = T_o_i.dot(T_i_f)
print(np.round(T_o_f,2))