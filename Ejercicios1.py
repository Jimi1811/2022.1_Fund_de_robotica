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

# Generalizar transf. homogenea T
d_vacio=np.array([[0,0,0]]).T
R_vacio=np.eye(3)

##############################################
# De la pag: Ejercicios1
##############################################

# Se ha cambiado la representacion
# p_A = Punto en A
# R_A_B = Matriz de rotacion que empieza en A y termina en B

# Pregunta 1
print(S_R_z(t)*S_R_y(p))
print((S_R_z(t)*S_R_y(p)).T)


""" 
# Pregunta 2
R_f_i = R_y(np.pi/2).dot(R_z(np.pi/4))
p = np.array([1,1,1]).T
p_final = (R_f_i).T.dot(p)
print(np.round(p_final,2))
"""
"""
# Pregunta 3
p_B=np.array([[2,4,5]]).T
R_A_B=np.array([
    [1,0,0],
    [0,0.6,0.8],
    [0,-0.8,0.6]
])
p_A = R_A_B.T.dot(b_p)
# a
print(np.round(p_A,3))
# b
print(np.round(R_A_B.dot(p_A),3))
"""

""" # Pregunta 4
a_R_b = np.array([
    [1,0,0],
    [0,1/2,-np.sqrt(3)/2],
    [0,np.sqrt(3)/2,1/2]
])

a_R_c = np.array([
    [0,0,-1],
    [0,1,0],
    [1,0,0]
])
b_R_c = a_R_b.T.dot(a_R_c)
print(np.round(b_R_c,2)) """

""" # Pregunta 5
M=np.array([
    [0.3536,-0.6124,0.7071],
    [0.9268,0.1268,-0.3536],
    [0.1298,0.7803,0.6124]
])
prop_R(M) """

""" # Pregunta 6
R=R_z(-np.pi/2).dot(R_x(np.pi/2))
print(np.round(R,2)) """
""" 
# Pregunta 7
R_z = S_R_z(t)
R_x = S_R_x(p)
R_y = S_R_y(bb)
Rf = R_x * R_z * R_y
print(Rf)
print(np.rad2deg(np.arcsin(-0.866)))
"""

"""
# Pregunta 8 
R_z=R_z(np.pi/6)
R_x=R_x(np.pi/2)
R_y=R_y(np.pi/180*50)
R_final=R_x.dot(R_z).dot(R_y)
print(np.round(R_final,3))
"""