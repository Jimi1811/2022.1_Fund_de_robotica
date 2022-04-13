import os 
os.system("clear") 

import numpy as np

B = np.array( [[10,20,30], [40,50,60]] )    # Creación de un arreglo bidimensional
# print("B =\n", B)
# print("Tamaño de B:", B.shape)   
# print("Elementos: B[0,0] =", B[0,0], ", B[0,1] =", B[0,1], ", B[1,0] =", B[1,0])

##################################################################################
#                       UNIDIMENSIONAL -> BIDIMENSIONAL
###################################################################################
# De arreglo unidimensional a vector fila o columna
x = np.array([10, 20, 30, 40])      # Arreglo unidimensional (4,)
# print("Tamaño de x:", x.shape)
# Conversión a vector columna 
x1 = x[:,None]                      # Equivalente a x.reshape(3,1) - VECTOR COLUMNA
# print("Tamaño del vector columna:", x1.shape)
# Conversión a vector fila
x2 = x[None,:]                      # Equivalente a x.reshape(1,3) o x[None] - VECTOR FILA
# print("Tamaño del vector fila:", x2.shape)

# De arreglo bidimensional (vector fila o columna) a arreglo unidimensional
v = np.array([[10, 20, 30]])
# print("\nTamaño de v:", v.shape)
# Conversión a arreglo unidimensional
v0 = v.flatten()
# print("Tamaño del arreglo unidimensional:", v0.shape)

###################################################################################
#                        MATRICES PREDETERMINADAS
###################################################################################
# A1 = np.zeros((2,2))   # Crea una matriz de tamaño (2 x 2) con ceros
# print("np.zeros((2,2)):\n", A1)

# A2 = np.ones((1,2))    # Crea una matriz de unos de tamaño (1 x 2)
# print("np.ones((1,2)):\n", A2)

# A3 = np.full((3,2), 7)  # Crea una matriz constante con "7"s de tamaño (3 x 2)
# print("np.full((3,2), 7):\n", A3)

# A4 = np.eye(3)         # Crea una matriz identidad de 3 x 3
# print("np.eye(3):\n", A4)

# A5 = np.random.random((2,4))  # Crea una matriz aleatoria de 2x4
# print("np.random.random((2,4)):\n", A5) 

###################################################################################
#                             Tipos de datos
###################################################################################

# x = np.array([1, 2])       # NumPy adivina el tipo de dato
# print("Tipo de dato de x:", x.dtype)

# y = np.array([1.0, 2.0])   # NumPy adivina el tipo de dato
# print("Tipo de dato de y:", y.dtype)             

# # En el siguiente arreglo se indica explícitamente el tipo de dato: int64
# z = np.array([1, 2], dtype=np.int64) 
# print("Tipo de dato de z:", z.dtype)   

###################################################################################
#                             Tipos de datos
###################################################################################

A = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print("Matriz A:\n", A)

# Sub-matriz
B = A[:2, 1:3]    # Toma las filas 0, 1 (excluyendo a 2), y las columnas 1, 2 (excluyendo a 3)
print("Submatriz B=A[:2, 1:3]:\n", B)

# Una submatriz es una vista de los mismos datos, así que modificarla modificará
# también la matriz inicial
B[1, 1] = 50     # B[1,1] contiene la misma información que A[1, 2]
print("Matriz B modificada:\n", B)
print("Matriz A resultante:\n", A)