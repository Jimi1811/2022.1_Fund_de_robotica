import os 
os.system("clear") 

import numpy as np

d_vacio=np.array([[0,0,0]]).T
R_vacio=np.eye(3)

######################################################
#               MATRIZ DE ROTACION R
######################################################

# PROPIEDADES PARA LA MATRIZ DE ROTACION R
def prop_R(R):
    # det(R) = 1
    det_R = np.linalg.det(R)
    # R*R' = I
    Ide_R = np.dot(R,R.T)
    # R' = R^-1
    Tra_R = R.T
    Inv_R = np.linalg.inv(R)

    print("Debe ser identidad: \n",np.round(Ide_R,decimals=2))
    print("Transpuesta: \n", np.round(Tra_R,decimals=2))
    print("Inversa: \n", np.round(Inv_R,decimals=2))
    print("Determinante: \n",np.round(det_R,decimals=2))


# ROTACIONES PURAS
    # ALREDEDOR DE X
def R_x(theta_x):
    # En rad
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
        ])
    return Rx

# ALREDEDOR DE Y
def R_y(theta_y):
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
    return Ry

# ALREDEDOR DE Z
def R_z(theta_z):
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0, 0, 1]
        ])
    return Rz

######################################################
#      MATRIZ DE TRANSFORMACION HOMOGENEA T
######################################################
"""
# TRANFORMACIONES PURAS
# 1. ROTACIONES PURAS
T_rot_x = np.eye(4)
T_rot_x[:3,:3]=Rx
# print(T_rot_x)

T_rot_y = np.eye(4)
T_rot_y[:3,:3]=Ry
# print(T_rot_y)

T_rot_z = np.eye(4)
T_rot_z[:3,:3]=Rz
# print(T_rot_z)

# 2. TRASLACIONES
# d=np.array([[1,2,3]]) 
d=d.T
dx=d[0]
dy=d[1]
dz=d[2]

T_tra =np.eye(4)
T_tra[:3,-1:]=d
# print(T_tra)

T_tra_x =np.eye(4)
T_tra_x[:1,-1:]=dx
# print(T_tra_x)

T_tra_y =np.eye(4)
T_tra_y[1:2,-1:]=dy
# print (T_tra_y)

T_tra_z =np.eye(4)
T_tra_z[2:3,-1:]=dz
# print(T_tra_z)
"""

def T(R,d):
    # T general
    T = np.eye(4)
    T[:3,:3]=R
    T[:3,-1:]=d
    # print(T)
    return T



""" T_rot_x=T(R_x(np.pi/2),d_vacio)
T_tra = T(R_vacio,d)
rpta=T_tra.dot(T_rot_x)
print(np.round(rpta,decimals=2))
 """