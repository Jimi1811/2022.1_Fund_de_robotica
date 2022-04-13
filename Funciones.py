######################################################
#                  CONFIGURACION
######################################################

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

    print("Determinante: \n",np.round(det_R,decimals=3))
    print("Debe ser identidad: \n",np.round(Ide_R,decimals=2))
    print("Transpuesta: \n", np.round(Tra_R,decimals=3))
    print("Inversa: \n", np.round(Inv_R,decimals=3))
    print("Inversa = Transpuesta : \n", (np.round(Tra_R,decimals=2) == np.round(Inv_R,decimals=2)))

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
# Propiedades de T:
def prop_T(T):
    # Inversa es distinto a la transpuesta
    Tra_T = T.T
    Inv_T = np.linalg.inv(T)
    print("Transpuesta: \n", np.round(Tra_T,decimals=3))
    print("Inversa: \n", np.round(Inv_T,decimals=3))
    print("Inversa != Transpuesta : \n", (np.round(Tra_T,decimals=2) != np.round(Inv_T,decimals=2)))
   

# Hallar T a partir de R y vector d
def T(R,d):
    # T general
    T = np.eye(4)
    T[:3,:3]=R
    T[:3,-1:]=d
    # print(T)
    return T


######################################################
#               R y T simbolizadas
######################################################

def S_R_x(ang):
    # En rad
    Rx = sp.Matrix([
        [1, 0, 0],
        [0, cos(ang), -sin(ang)],
        [0, sin(ang), cos(ang)]
        ])
    return Rx
# R_y
def S_R_y(ang):
    Ry = sp.Matrix([[cos(ang), 0, sin(ang)],
                    [0, 1, 0],
                    [-sin(ang), 0, cos(ang)]])
    return Ry
# R_z
def S_R_z(ang):
    Rz = sp.Matrix([[cos(ang), -sin(ang), 0],
                   [sin(ang), cos(ang), 0],
                   [0,0,1]])
    return Rz

######################################################
######################################################
#        PARAMETRIZACIONES PARA HALLAR R
######################################################
######################################################

######################################################
#                      ZYZ 
######################################################

def ZYZ_R(phi1,phi2,phi3):
    R=np.array([
        [np.cos(phi1)*np.cos(phi2)*np.cos(phi3)-np.sin(phi1)*np.sin(phi3) , -np.sin(phi1)*np.cos(phi3)-np.cos(phi1)*np.cos(phi2)*np.sin(phi3) , np.cos(phi1)*np.sin(phi2)],
        [np.sin(phi1)*np.cos(phi2)*np.cos(phi3)+np.cos(phi1)*np.sin(phi3) , np.cos(phi1)*np.cos(phi3)-np.sin(phi1)*np.cos(phi2)*np.sin(phi3) , np.sin(phi1)*np.sin(phi2)],
        [-np.sin(phi2)*np.cos(phi3) , np.sin(phi2)*np.sin(phi3) , np.cos(phi2)] 
    ])

    return R

######################################################
#                  EJE / ANGULO 
######################################################
# De la formula de Rodrigues
# - Cuando tienes R y necesitas vector y angulo 
def eje_angulo(R):
    c = (R[0,0]+R[1,1]+R[2,2]-1.0)/2.0
    s = np.sqrt((R[1,0]-R[0,1])**2+(R[2,0]-R[0,2])**2+(R[2,1]-R[1,2])**2)/2.0
    th = np.arctan2(s,c)
    u = 1.0/(2.*sin(th))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return u,th

######################################################
#                 CUATERNION UNITARIO
######################################################

# Cuaternion Q=(w,ex,ey,ez) a partir de R
def Q_vector(R):
    w = 0.5*np.sqrt(1+R[0,0]+R[1,1]+R[2,2])
    ex = 1/(4*w)*(R[2,1]-R[1,2])
    ey = 1/(4*w)*(R[0,2]-R[2,0])
    ez = 1/(4*w)*(R[1,0]-R[0,1])
    return np.array([w, ex, ey, ez])

# Matriz de rotacion R a partir de Q
def Q_R_verif(Q):
    w = Q[0]; ex = Q[1]; ey = Q[2]; ez = Q[3]
    R = np.array([
        [2*(w**2+ex**2)-1,   2*(ex*ey-w*ez),    2*(ex*ez+w*ey)],
        [  2*(ex*ey+w*ez), 2*(w**2+ey**2)-1,    2*(ey*ez-w*ex)],
        [  2*(ex*ez-w*ey),   2*(ey*ez+w*ex), 2*(w**2+ez**2)-1]
    ])
    return R

# Representacion con eje y angulo a partir de Q
def Q_eje_angulo(Q):
    w = Q[0]; ex = Q[1]; ey = Q[2]; ez = Q[3]
    e = np.array([ex, ey, ez])
    u = e / np.linalg.norm(e)    # Eje

    th = 2 * np.arctan2(np.linalg.norm(e), w)    # Ángulo en radianes
    # th_deg = th/np.pi*180    # En grados
    return u,th