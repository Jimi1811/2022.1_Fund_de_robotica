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

##############################################
# De la pag: Ejercicios1
##############################################

# Pregunta 1
""" 
R_y=R_y(np.pi/2)
R_z=R_z(np.pi/4)
R=R_y.dot(R_z)
print("R: \n", np.round(R,3))
u,th=eje_angulo(R)
print(u,np.rad2deg(th)) """

""" 
# Pregunta 2
R=ZYZ_R(np.pi/2,np.pi/6,np.pi/4)
print(np.round(R,3))
R1=R_z(np.pi/2).dot(R_y(np.pi/6)).dot(R_z(np.pi/4))
print(np.round(R1,3))
"""
""" 
# Pregunta 3
b_T_a = np.array([
    [1,0,0,2],
    [0,0.6,0.8,-1],
    [0,-0.8,0.6,1],
    [0,0,0,1]
])
b_p=np.array([2,4,5,1]).T

a_p = np.linalg.inv(b_T_a).dot(b_p)
a_p=a_p[0:3].flatten() # elliminando el último elemento
print(np.round(a_p,3)) """

""" 
# Pregunta 4
R = np.array([
    [0.7905 , -0.3864 , 0.4752],
    [0.6046 , 0.3686 , -0.7061],
    [0.0977 , 0.8455 , 0.525]
])
Q = Q_vector(R)
print(np.round(Q,3)) """

""" # Pregunta 5
R = R_z(0).dot(R_x(np.pi/2)).dot(R_z(0))
Q = Q_vector(R)
print(np.round(Q,3))
 """
""" 
# Pregunta 6
T1 = np.array([
    [         -1/2, 0, -np.sqrt(3)/2,  1],
    [            0, 1,             0, -2],
    [-np.sqrt(3)/2, 0,          -1/2, -1],
    [            0, 0,             0,  1]]) 

T2 = np.array([[         -1/2, 0, -np.sqrt(3)/2,  0],
    [            0, 1,             0,  1],
    [ np.sqrt(3)/2, 0,          -1/2,  0],
    [            0, 0,             0,  1]]) 

T3 = np.array([[         -1/2, 0, -np.sqrt(3)/2,  1],
    [            0, 1,             0,  0],
    [ np.sqrt(3)/2, 0,          -1/2, -1],
    [            0, 0,             0,  1]])

# Para T1:
# prop_T(T1)
# prop_R(T1[:3,:3])
# # Para T2:
# prop_T(T2)
# prop_R(T2[:3,:3])
# # Para T3:
# prop_T(T3)
# prop_R(T3[:3,:3])

# Cuaternion
Q=Q_vector(T2[:3,:3])
# print(np.round(Q,3))

# Eje y angulo
u, th = eje_angulo(T2[:3,:3])
print(np.round(np.rad2deg(th),3))
print(u)
 """
""" 
# Pregunta 7 
# T del sistema 2 en funcion de sistema 1
T_1_2 = T(R_z(np.pi/2),np.array([[0,0.4,0.2]]).T)
# print(np.round(T_1_2,2))
# T del sistema 1 en funcion del sistema 3
T_3_1 = T(R_x(np.pi).dot(R_z(-np.pi/2)),np.array([[-0.5,0.5,2]]).T) 
# print(np.round(T_3_1,2))
T_3_2 = T_3_1.dot(T_1_2)
# T de 2 en funcion de 3 : T_3_2
# print(np.round(T_3_2,2))

# T de 2 en funcion de 0: T_0_2
# T_0_2 = T_0_1 * T_1_2
T_0_1 = T(R_vacio,np.array([[0,1,1]]).T)
T_0_2 = T_0_1.dot(T_1_2)
print(np.round(T_0_2,2))
"""
""" 
# Pregunta 8
R_final = sp.simplify(S_R_z(p1)*S_R_x(p2)*S_R_z(p3))
print(R_final)
"""

""" 
# Pregunta 9
R = np.array([[ 0.7905,  0.6046, 0.0977],
    [-0.3864,  0.3686, 0.8455],
    [ 0.4752, -0.7061, 0.5250]])

Q = Q_vector(R)
# a)
# print(Q)
# b)
u,th = Q_eje_angulo(Q)
# print(u)
# print(np.round(th/np.pi*180,2))
# c)
R_inv=R.T
Q_nuevo=Q_vector(R_inv)
print(Q_nuevo)
"""

""" # Pregunta 10
R = R_x(np.pi/2)
Q = Q_vector(R)
print(np.round(Q,2))
"""
""" 
# Pregunta 11
R = R_z(10/180*np.pi).dot(R_y(70/180*np.pi)).dot(R_x(30/180*np.pi))
print(np.round(R,2))
# Primer conjunto de ángulos
phi2 = np.rad2deg( np.arctan2(-R[2,0], np.sqrt(R[2,2]**2+R[2,1]**2)) )
phi3 = np.rad2deg( np.arctan2(R[2,1]/np.cos(phi2), R[2,2]/np.cos(phi2)) )
phi1 = np.rad2deg( np.arctan2(R[1,0]/np.cos(phi2), R[0,0]/np.cos(phi2)) )
print(np.array([phi1, phi2, phi3]))

# Segundo conjunto de ángulos
phi2 = np.rad2deg( np.arctan2(-R[2,0], -np.sqrt(R[2,2]**2+R[2,1]**2)) )
phi3 = np.rad2deg( np.arctan2(R[2,1]/np.cos(phi2), R[2,2]/np.cos(phi2)) )
phi1 = np.rad2deg( np.arctan2(R[1,0]/np.cos(phi2), R[0,0]/np.cos(phi2)) )
print(np.array([phi1, phi2, phi3]))
"""
"""
# Pregunta 12
# T_2_1, T_0_3, T_0_1
T_0_1 = T(R_x(np.pi/2).dot(R_z(np.pi/2)),np.array([[3,0,0]]).T)
T_0_2 = T(R_y(np.pi/2).dot(R_x((90+53)/180*np.pi)),np.array([[0,4,4]]).T)
T_0_3 = T(R_y(-np.pi/2).dot(R_x((180+37)/180*np.pi)),np.array([[0,4,0]]).T)

T_2_1 = np.linalg.inv(T_0_2).dot(T_0_1)

print("T_2_1: \n",np.round(T_2_1,1))
print("T_0_3: \n",np.round(T_0_3,1))
print("T_0_1: \n",np.round(T_0_1,1))

# T_2_3
T_2_3 = np.linalg.inv(T_0_2).dot(T_0_3)
print("T_2_3: \n",np.round(T_2_3,1)) """