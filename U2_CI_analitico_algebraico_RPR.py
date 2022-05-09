# Para limpiar terminal
import os 
os.system("clear") 

import numpy as np

def cinematica_directa_RPR(q1, q2, q3, a3, d1):
    q13 = q1+q3
    T = np.array([[np.cos(q13), 0,  np.sin(q13), a3*np.cos(q13)+q2*np.sin(q1)],
                  [np.sin(q13), 0, -np.cos(q13), a3*np.sin(q13)-q2*np.cos(q1)],
                  [          0, 1,            0,                           d1],
                  [          0, 0,            0,                            1]])
    return T

def cinematica_inversa_RPR(Tdes, a3, d1):
    nx = Tdes[0,0]
    ny = Tdes[1,0]
    ax = Tdes[0,2]
    px = Tdes[0,3]
    py = Tdes[1,3]
    # Valores articulares calculados
    q1 = np.arctan2(px-a3*nx, a3*ny-py)
    q3 = np.arctan2(ax, nx) - q1
    q2 = px*np.sin(q1) - py*np.cos(q1) - a3*(nx*np.sin(q1)-ny*np.cos(q1))
    # Resultado
    return (q1, q2, q3)


# Prueba de la cinemática inversa
# -------------------------------

# Parámetros del robot
a3 = 1; d1 = 1
# Generación de una matriz a partir de la cinemática directa
Tdes = cinematica_directa_RPR(0.5, 1.0, 1.35, 1, 1)

# Uso de la cinemática inversa
Q = cinematica_inversa_RPR(Tdes, a3, d1)
print("Valores articulares resultantes:", Q)