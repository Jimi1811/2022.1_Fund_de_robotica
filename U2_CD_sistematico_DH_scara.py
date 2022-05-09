from Funciones import *  
from Funciones_simbolico import *  
# Para limpiar terminal
import os 
os.system("clear") 

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

# ------------------------------------------------------

# Transformaciones homogéneas
T01 = S_T_DH(    l1,    sp.pi+q1, l2,     0)
T12 = S_T_DH(     0, -sp.pi/2+q2, l3,     0)
T23 = S_T_DH(-l4+q3,           0,  0,     0)
T34 = S_T_DH(     0,  sp.pi/2+q4,  0, sp.pi)

# Transformación homogénea final
Tf = sp.simplify(T01*T12*T23*T34)

# Mostrar las transformaciones homogéneas (display funciona con IPython)
print("T01:\n", T01)
print("T12:\n", T12)
print("T23:\n", T23)
print("T34:\n", T34)
print("T04:\n", Tf)

# Valor cuando todos los ángulos son cero
print("T04 cuando q=(0,0,0,0):")
Tf_q = Tf.subs([ (q1,0), (q2,0), (q3,0), (q4,0)])
print("T04 reemplazado:\n", Tf_q)

