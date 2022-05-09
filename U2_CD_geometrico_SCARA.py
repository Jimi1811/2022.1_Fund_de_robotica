from Funciones import * 
from Funciones_simbolico import *  

# Para limpiar terminal
import os 
os.system("clear") 

# Realizar ploteo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# ---------------------------------------------------------

# Transformaciones con respecto al sistema anterior
T_0_1 = S_T_tra(0,0,l1)*S_T_rot_z(sp.pi+q1)
T_1_2 = S_T_tra(l2,0,0)*S_T_rot_z(-sp.pi/2+q2)
T_2_3 = S_T_tra(l3,0,0)
T_3_4 = S_T_tra(0,0,-l4+q3)*S_T_rot_z(sp.pi/2+q4)

# Transformación del eslabón 4 con respecto a la base (0)
T_0_4 = sp.simplify(T_0_1*T_1_2*T_2_3*T_3_4)

# Mostrar las transformaciones homogéneas (display funciona con IPython)
print("T01:\n", T_0_1)
print("T12:\n", T_1_2)
print("T23:\n", T_2_3)
print("T34:\n", T_3_4)
print("T04:\n", T_0_4)

# Transformación del efector final con respecto al sistema 4
T_4_e = S_T_rot_x(sp.pi)

# Transformación del efector final con respecto a la base (0)
T_0_e = sp.simplify(T_0_4*T_4_e)
print("T0e: \n", T_0_e)

# Valor cuando todos los ángulos son cero
T_0_e_q = T_0_e.subs([ (q1,0), (q2,0), (q3,0), (q4,0)])
print("T0e cuando q=(0,0,0,0):\n", T_0_e_q)

# -----------------------------------------------------------------

# Ejemplo de cálculo de la cinemática directa
l1 = 1.0                               # Longitud eslabón 1
l2 = 1.0                               # Longitud eslabón 2
l3 = 1.0                               # Longitud eslabón 3 
l4 = 0.5
q = [np.deg2rad(0), np.deg2rad(0), 0, np.deg2rad(0)]    # Valores articulares

# Cinemática directa
Te, T = CD_scara(q, l1, l2, l3, l4)   # Cinemática directa

# Mostrar el resultado
print("Efector final con respecto a la base cuando q1={}, q2={}, q3={}, q4={}".format(np.rad2deg(q[0]), np.rad2deg(q[1]), 
                                                                                      q[2], np.rad2deg(q[3])))
print(np.round(Te,4))

# PLOTEO
def graph_scara(q, l1, l2, l3, l4, k=0.4):
    """ Grafica el robot según la configuración articular. Las entradas son los valores articulares, 
    las longitudes de los eslabones y un factor para el tamaño con que se muetra los sistemas de referencia
    """
    # Cálculo de la cinemática directa
    Te, T = CD_scara(q, l1, l2, l3, l4)
    # Borrar el gráfico
    plt.clf()
    ax = plt.axes(projection='3d')
    # Nombres para los ejes
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    # Transformaciones homogéneas con respecto a la base (ej. T2 es {2} con respecto a {0})
    T1 = T[0]; T2 = T[1]; T3 = T[2]; T4 = T[3]
    # Cuerpo del robot
    ax.plot([0, T1[0,3]], [0, T1[1,3]], [0, T1[2,3]], linewidth=3, color='k')
    ax.plot([T1[0,3], T2[0,3]], [T1[1,3], T2[1,3]], [T1[2,3], T2[2,3]], linewidth=3, color='k')
    ax.plot([T2[0,3], T3[0,3]], [T2[1,3], T3[1,3]], [T2[2,3], T3[2,3]], linewidth=3, color='k')
    ax.plot([T3[0,3], T4[0,3]], [T3[1,3], T4[1,3]], [T3[2,3], T4[2,3]], linewidth=3, color='k')
    # Puntos en las articulaciones
    ax.scatter(0, 0, 0, color='g', s=50)
    # "Cilindros" para representar la dirección de las articulaciones
    ax.plot([T1[0,3], T1[0,3]], [T1[1,3], T1[1,3]], [T1[2,3]-0.1, T1[2,3]+0.1], linewidth=10, color='g')
    ax.plot([T2[0,3], T2[0,3]], [T2[1,3], T2[1,3]], [T2[2,3]-0.1, T2[2,3]+0.1], linewidth=10, color='g')
    ax.plot([T3[0,3], T3[0,3]], [T3[1,3], T3[1,3]], [T3[2,3]-0.1, T3[2,3]+0.1], linewidth=10, color='g') 
    ax.plot([T4[0,3], T4[0,3]], [T4[1,3], T4[1,3]], [T4[2,3]-0.05, T4[2,3]+0.05], linewidth=10, color='g')    
    # Efector final (definido por 4 puntos)
    p1 = np.array([0, 0.1, 0, 1]); p2 = np.array([0, 0.1, 0.2, 1])
    p3 = np.array([0, -0.1, 0, 1]); p4 = np.array([0, -0.1, 0.2, 1])
    p1 = Te.dot(p1); p2 = Te.dot(p2); p3 = Te.dot(p3); p4 = Te.dot(p4)
    # Sistema de referencia del efector final (con respecto al sistema 0)
    ax.plot([Te[0,3],Te[0,3]+k*Te[0,0]], [Te[1,3],Te[1,3]+k*Te[1,0]], [Te[2,3],Te[2,3]+k*Te[2,0]], color='r')
    ax.plot([Te[0,3],Te[0,3]+k*Te[0,1]], [Te[1,3],Te[1,3]+k*Te[1,1]], [Te[2,3],Te[2,3]+k*Te[2,1]], color='g')
    ax.plot([Te[0,3],Te[0,3]+k*Te[0,2]], [Te[1,3],Te[1,3]+k*Te[1,2]], [Te[2,3],Te[2,3]+k*Te[2,2]], color='b')
    # Sistema de referencia de la base (0)
    ax.plot([0,k], [0,0], [0,0], color='r')
    ax.plot([0,0], [0,k], [0,0], color='g')
    ax.plot([0,0], [0,0], [0,k], color='b')
    # Gráfico del efector final
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color='b', linewidth=3)
    ax.plot([p3[0],p4[0]], [p3[1],p4[1]], [p3[2],p4[2]], color='b', linewidth=3)
    ax.plot([p1[0],p3[0]], [p1[1],p3[1]], [p1[2],p3[2]], color='b', linewidth=3)
    # Punto de vista
    ax.view_init(elev=25, azim=45)
    # Límites para los ejes
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0,1.2)
    plt.show()

# ------------------------------------------------------------------------------

ax = graph_scara([0,0,0,0], l1, l2, l3, l4)

""" # Se abrirá una nueva ventana donde se visualizará el robot
for i in range(40):
    q = [np.deg2rad(i), np.deg2rad(0.5*i), 0.005*i, np.deg2rad(0.5*i)]    # En grados
    graph_scara(q, l1, l2, l3, l4)

    plt.pause(0.01)   """