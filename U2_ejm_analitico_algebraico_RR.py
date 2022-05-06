# Para limpiar terminal
import os 
os.system("clear") 

import numpy as np
# import matplotlib.pyplot as plt

# Función que realiza el cálculo de la cinemática inversa (calculado usando geometría)
def cinematica_inversa_RR_alg(x, y, L1, L2):
    c2 = (x**2+y**2-L1**2-L2**2)/(2*L1*L2)
    s2a =  np.sqrt(1-c2**2)
    s2b = -np.sqrt(1-c2**2)
    # Dos valores para q2:
    q2a = np.arctan2(s2a, c2)
    q2b = np.arctan2(s2b, c2)
    # Solución 1 para q1 (usando el primer valor posible de q2)
    A = np.array([[L1+L2*c2,  -L2*s2a],
                  [  L2*s2a, L1+L2*c2]])
    v = np.dot( np.linalg.inv(A), np.array([x,y]) )
    c1 = v[0]; s1 = v[1]
    q1a = np.arctan2(s1, c1)
    # Solución 2 para q1 (usando el segundo valor posible de q2)
    A = np.array([[L1+L2*c2,  -L2*s2b],
                  [  L2*s2b, L1+L2*c2]])
    v = np.dot( np.linalg.inv(A), np.array([x,y]) )
    c1 = v[0]; s1 = v[1]
    q1b = np.arctan2(s1, c1)
    # Solución 2:
    # Retornar ambas soluciones
    return ((q1a, q2a), (q1b, q2b))
    #return (q1a, q2a)
    
    
# Prueba de la cinemática inversa
# -------------------------------

L1 = 1.0; L2 = 1.0  # Longitudes fijas
xdes = 1.2          # Valor deseado en x
ydes = 1.2          # Valor deseado en y

# Cálculo de la cinemática inversa
Q = cinematica_inversa_RR_alg(xdes, ydes, L1, L2)

print("Solución 1:", np.round(Q[0], 4))
print("Solución 2:", np.round(Q[1], 4))