######################################################
#           LIBRERIAS 
######################################################
from Funciones import *

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

######################################################
#           Pregunta 1
######################################################

# --------------------- consigna a ------------------

# T de 0 a 2
T_0_2 = T(R_y(-np.pi/4),np.array([[0, 0, 0.3]]).T)
# print(np.round(T_0_2,3))

# T de 2 a 4
R1 = np.array([
    [0, 0, 1],
    [0, -1, 0],
    [1, 0, 0]
])

T_2_4 = T_tra_z(0.75).dot(T_rot_y(np.pi/4)).dot(T(R1,d_vacio))
# print(np.round(T_2_4,3))

# --------------------- consigna b ------------------

# 1. T desde 4 a 0
# 1.1 T de 0 a 4
T_0_4 = T_0_2.dot(T_2_4)
# 1.2 T de 4 a 0 
T_4_0 = np.linalg.inv(T_0_4)
# print(np.round(T_4_0,3))

# 2. Angulos de euler XZX
R = T_4_0[0:3,0:3]
# 2.1 Obteniendo por simbolico la matriz R
R_XZX = S_R_x(p1)*S_R_z(p2)*S_R_x(p3)

#print(R_XZX)
# [[cos(p2),                                   -sin(p2)*cos(p3),                            sin(p2)*sin(p3)],
#  [sin(p2)*cos(p1), -sin(p1)*sin(p3) + cos(p1)*cos(p2)*cos(p3), -sin(p1)*cos(p3) - sin(p3)*cos(p1)*cos(p2)], 
#  [sin(p1)*sin(p2),  sin(p1)*cos(p2)*cos(p3) + sin(p3)*cos(p1), -sin(p1)*sin(p3)*cos(p2) + cos(p1)*cos(p3)]]

# 2.2 Sacando angulos con r11, r21, r31, r21, r31
phi_p = XZX_ang_pos(R)

phi_n = XZX_ang_neg(R)

# print("Para raíz positiva: \n", phi_p)
# print("Para raíz negativa: \n", phi_n) 

# --------------------- consigna c ------------------
Q_XZX = Q(R)
print(R) 

######################################################
#           Pregunta 2
######################################################
""" 
# 1. R de Io a B
# 1.1 Simbolico de ZYX
S_R_ZYX = S_R_z(p1)*S_R_y(p2)*S_R_x(p3)
# print(S_R_ZYX)
# Matriz simbolica
#  [cos(p1)*cos(p2), -sin(p1)*cos(p3) + sin(p2)*sin(p3)*cos(p1), sin(p1)*sin(p3) + sin(p2)*cos(p1)*cos(p3)],
#  [sin(p1)*cos(p2),  sin(p1)*sin(p2)*sin(p3) + cos(p1)*cos(p3), sin(p1)*sin(p2)*cos(p3) - sin(p3)*cos(p1)], 
#  [-sin(p2),                                   sin(p3)*cos(p2),                           cos(p2)*cos(p3)]
# 1.2 En una funcion cambiando para operar
R_B_Io = ZYX_R(np.deg2rad(35),np.deg2rad(-35),np.deg2rad(10))

print(np.round(R_B_Io,3))
# 2. R de B a If
R_B_If = np.array([
    [0, np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60))],
    [0, np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60))],
    [-1, 0, 0]
])
print(np.round(R_B_If,3))

# 3. R relativa
R_Io_If = R_B_Io.T.dot(R_B_If)

# print(np.round(R_Io_If,3))

# 4. Pasarlo a eje/angulo

# print("Raiz positiva: \n",eje_angulo_pos(R_Io_If))
# print("Raiz negativa: \n",eje_angulo_neg(R_Io_If))
 """

######################################################
#           Pregunta 3
######################################################

# --------------------- consigna a ------------------
# 1. distancia

# D = np.sqrt(1+1+4)
# print(D)
