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

# ----------------------------------------------------------
# Pregunta 1
# ----------------------------------------------------------
""" 
# Matrices de rotación
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

# b)
R_o_f = R_i_o.T.dot(R_i_f)
# print(np.round(R_o_f,2))
u1,th1 = eje_angulo_pos(R_o_f)
# print("vector u: ", u1)
# print("Angulo en deg: ", np.round(th1/np.pi*180,2))
u2,th2 = eje_angulo_neg(R_o_f)
# print("vector u: ", u2)
# print("Angulo en deg: ", np.round(th2/np.pi*180,2))

# c)
T_i_f = T(R_i_f,np.array([[2,4,-3]]).T)
# print(np.round(T_i_f,2))

T_o_i = np.linalg.inv(T(R_i_o,d_vacio))
# print(np.round(T_o_i,2))

T_o_f = T_o_i.dot(T_i_f)
print(np.round(T_o_f,2)) 
 """

# ----------------------------------------------------------
# Pregunta 2
# ----------------------------------------------------------

Q = np.array([0.939 , 0.093 , -0.224 , 0.242])

# ----------------------------------------------------------
# a) 
# 1. Por formula, se saca R a partir de Q
R_Q = Q_R(Q)
# print(R_Q)

# 2. Encontrando los valores de R para YXY con simbolos
S_R_YXY = S_R_y(p1)*S_R_x(p2)*S_R_y(p3)
# print(S_R_YXY)

# R_YXY= sp.Matrix([
#     [-sin(p1)*sin(p3)*cos(p2) + cos(p1)*cos(p3), sin(p1)*sin(p2), sin(p1)*cos(p2)*cos(p3) + sin(p3)*cos(p1)], 
#     [sin(p2)*sin(p3),                                    cos(p2),                          -sin(p2)*cos(p3)], 
#     [-sin(p1)*cos(p3) - sin(p3)*cos(p1)*cos(p2), sin(p2)*cos(p1), -sin(p1)*sin(p3) + cos(p1)*cos(p2)*cos(p3)]
#     ])

YXY_angulos_pos = YXY_ang_pos(R_Q)
print(YXY_angulos_pos)

YXY_angulos_neg = YXY_ang_neg(R_Q)
print(YXY_angulos_neg)

# ----------------------------------------------------------
# b) 

R_YXY_0= sp.Matrix([
    [-sin(p1)*sin(p3) + cos(p1)*cos(p3), 0, sin(p1)*cos(p3) + sin(p3)*cos(p1)], 
    [0,                                    1,                          0], 
    [-sin(p1)*cos(p3) - sin(p3)*cos(p1), 0, -sin(p1)*sin(p3) + cos(p1)*cos(p3)]
    ])

R_YXY_pi= sp.Matrix([
    [sin(p1)*sin(p3) + cos(p1)*cos(p3), 0, sin(p1)*cos(p2)*cos(p3) + sin(p3)*cos(p1)], 
    [0,                                   -1,                          0], 
    [-sin(p1)*cos(p3) + sin(p3)*cos(p1), 0, -sin(p1)*sin(p3) - cos(p1)*cos(p3)]
    ])
 
# ----------------------------------------------------------
# Pregunta 3
# ----------------------------------------------------------
""" 
# Dato
T_B_final = np.array([
    [0.707, 0, 0.707, 70],
    [0,     -1,    0, -40],
    [0.707, 0, -0.707, 80],
    [0,      0,     0,  1],
])

T_E_C2 = np.array([
    [0, -1, 0, 5],
    [1, 0, 0, -10],
    [0, 0, 1, -4],
    [0, 0, 0, 1]
])

T_C2_C1 = np.array([
    [1, 0, 0, 20],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 1. Encontrando T de efector final respecto a su referencia inicial
T_E_final = T_tra_y(18).dot(T_rot_y(-np.pi/4)).dot(T_tra_z(8))
# 2. T de E respecto a B
T_B_E = T_B_final.dot(np.linalg.inv(T_E_final))

# 3. T de C1 respecto a B
T_B_C1 = T_B_E.dot(T_E_C2).dot(T_C2_C1)

print(np.round(T_B_C1,3))

"""

# ----------------------------------------------------------
# Pregunta 4
# ----------------------------------------------------------

r1, r2, r3 = sp.symbols("r1 r2 r3")
R = sp.Matrix([
    [0.91, 0.409,   r1],
    [-0.377, 0.886, r2],
    [-0.174, 0.22 , r3]
])
print(R*R.T)
# [r1**2 + 0.995381, r1*r2 + 0.019304, r1*r3 - 0.06836], 
# [r1*r2 + 0.019304, r2**2 + 0.927125, r2*r3 + 0.260518], 
# [r1*r3 - 0.06836, r2*r3 + 0.260518, r3**2 + 0.078676]
