# Para limpiar terminal
import os 
os.system("clear") 

import numpy as np
import matplotlib.pyplot as plt


def cinematica_directa_RR(q1, q2, L1, L2):
    x = L1*np.cos(q1) + L2*np.cos(q1+q2)
    y = L1*np.sin(q1) + L2*np.sin(q1+q2)
    return (x,y)

# Función que realiza el cálculo de la cinemática inversa (calculado usando el método de Newton)
def cinematica_inversa_RR_Gradiente(Xdeseado, Qinicial, L1, L2, alfa, max_iter, epsilon):
    # Es importante usar "copy" para no sobrescribir el valor original
    q = Qinicial.copy()
    # Almacenamiento del error
    ee = []
    # Bucle principal
    for i in range(max_iter):
        q1 = q[0]; q2 = q[1]
        J = np.array([[-L1*np.sin(q1)-L2*np.sin(q1+q2), -L2*np.sin(q1+q2)],
                      [ L1*np.cos(q1)+L2*np.cos(q1+q2),  L2*np.cos(q1+q2)]])
        f = np.array([L1*np.cos(q1)+L2*np.cos(q1+q2), L1*np.sin(q1)+L2*np.sin(q1+q2)])
        e = Xdeseado-f
        q = q + alfa*np.dot(J.T, e)
        # Norma del error
        enorm = np.linalg.norm(e)
        # print("Error en la iteración {}: {}".format(i, np.round(enorm,4)))
        ee.append(enorm)    # Almacena los errores
        # Condición de término
        if (np.linalg.norm(e) < epsilon):
            break
    return q, ee


# Prueba de la cinemática inversa
# -------------------------------

# Parámetros del robot
L1 = 1; L2 = 1
# Valor articular inicial
qinit  = np.array([0.5, 0.5])

# Valor x,y deseado (en el espacio cartesiano)
xd = np.array([1.2, 1.2])
# Hiperparámetros
epsilon = 1e-4         # Condición para el término
max_iteraciones = 100  # Máximo número de iteraciones
alfa=0.5               # Paso

# Cinemática Inversa
q, e = cinematica_inversa_RR_Gradiente(xd, qinit, L1, L2, alpha, max_iteraciones, epsilon)
print("Valores articulares obtenidos:", np.round(q,4))

# Verificación usando la cinemática directa
print("Coordenada x,y obtenida:", np.round(cinematica_directa_RR(q[0], q[1], L1, L2), 4))

# Gráfico del error (debe ser decreciente)
plt.plot(e,'b')
plt.plot(e,'b.')
plt.title("Evolución del error"); plt.grid()
plt.xlabel("Número de iteraciones"); plt.ylabel("Norma del error");

plt.show()