######################################################
#                      LIBRERIAS
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
#                      PREGUNTA 1
######################################################

""" # ------------------------- a ------------------------
# Transformacion desde la matriz fija F a la posicion y orientacion final del submarino
T_F_final = T_tra_y(2.1).dot(T_tra_x(1.5)).dot(T(R_z(np.pi/4),np.array([[0.15,-1.54,2.1]]).T)).dot(T_rot_y(np.deg2rad(20))).dot(T_tra_x(1.1))
# print(np.round(T_F_final,3))

# ------------------------- b ------------------------
# Trans. del submarino respecto al barco
T_F_B = np.array([
    [-1, 0, 0, 3],
    [0, 0, 1, 2.5],
    [0, 1, 0, 0.1],
    [0, 0, 0,   1]
])

T_B_final = np.linalg.inv(T_F_B).dot(T_F_final)

print(np.round(T_B_final,3))
 """

######################################################
#                      PREGUNTA 2
######################################################
""" 
# Matriz de rotación
R = np.array([
    [0.527, -0.574, 0.628],
    [0.369,  0.819, 0.439],
    [-0.766,     0, 0.643]
])

# ------------------------- a ------------------------
# 1. Generando 1 cuaternión 
# Q1 = Q(R)
# print(np.round(Q1,2))

# ------------------------- b ------------------------
# 1. Se saca la matriz con simbolicos de la secuencia ZYX
S_R_ZYX = S_R_z(p1)*S_R_y(p2)*S_R_x(p3)
# print(S_R_ZYX)
# Matriz simbolica

#  [cos(p1)*cos(p2), -sin(p1)*cos(p3) + sin(p2)*sin(p3)*cos(p1), sin(p1)*sin(p3) + sin(p2)*cos(p1)*cos(p3)],
#  [sin(p1)*cos(p2),  sin(p1)*sin(p2)*sin(p3) + cos(p1)*cos(p3), sin(p1)*sin(p2)*cos(p3) - sin(p3)*cos(p1)], 
#  [-sin(p2),                                   sin(p3)*cos(p2),                           cos(p2)*cos(p3)]

# 2. Sacando angulos con r11, r21, r31, r32, r33
phi_p = ZYX_ang_pos(R)

phi_n = ZYX_ang_neg(R)

print("Para raíz positiva: \n", phi_p)
print("Para raíz negativa: \n", phi_n) """


######################################################
#                      PREGUNTA 3
######################################################

""" # 1. T de efector final a cuerpo rigido
T_F_C = T(R_vacio,np.array([[0.5, 0.4, 0.6]]).T)
# 2. Encontrando la matriz de rotacion respecto al eje
th = np.deg2rad(40)
u = np.array([1.5, 1.5, 0.7])
u = u/np.linalg.norm(u)

R_eje_angulo = eje_angulo_R(u,th)
# R_eje_angulo = np.array([
#     [0.8715, -0.0959, 0.4809],
#     [0.3069, 0.8715, -0.3823],
#     [-0.3824, 0.4809, 0.7890],
#     ])
# print(R_eje_angulo)
# 3. Usando el metodo ROLL, PITCH, YAW, o ZYX
phi_p = ZYX_ang_pos(R_eje_angulo)
phi_n = ZYX_ang_neg(R_eje_angulo)

print("Para raíz positiva: \n", phi_p)
print("Para raíz negativa: \n", phi_n)

# b)
print(np.round(Q(R_eje_angulo),3)) """