from Funciones import *  
from Funciones_simbolico import *  

# Para limpiar terminal
import os 
os.system("clear") 

# Realizar graficos
import matplotlib.pyplot as plt

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
#              Polinomio cubico
# ------------------------------------------------------

""" A = np.array([[2**3, 2**2, 2, 1],
              [4**3, 4**2, 4, 1],
              [3*2**2, 2*2, 1, 0],
              [3*4**2, 2*4, 1, 0]])
b = np.array([10, 60, 0, 0]).T

# Se encuentran los parametros
x = np.linalg.inv(A).dot(b)
a3=x[0]; a2=x[1]; a1=x[2]; a0=x[3]

print(x)

t = np.linspace(2, 4, 100)
# Posición
q = a3*t**3 + a2*t**2 + a1*t + a0
# Velocidad
dq = 3*a3*t**2 + 2*a2*t + a1
# Aceleración
ddq = 6*a3*t + 2*a2

plt.figure(figsize=(10,10))
plt.subplot(3,1,1); plt.plot(t,q); plt.grid(); plt.title('posición'); plt.ylabel("q [°]")
plt.subplot(3,1,2); plt.plot(t,dq); plt.grid(); plt.title('velocidad'); plt.ylabel("dq [°/s]")
plt.subplot(3,1,3); plt.plot(t,ddq); plt.grid(); plt.title('aceleración'); plt.ylabel("ddq [°/$s^2$]");
plt.show()
 """

# ------------------------------------------------------
#              Polinomio cubico
# ------------------------------------------------------
""" A = np.array([[2**5, 2**4, 2**3, 2**2, 2, 1],
              [4**5, 4**4, 4**3, 4**2, 4, 1],
              [5*2**4, 4*2**3, 3*2**2, 2*2, 1, 0],
              [5*4**4, 4*4**3, 3*4**2, 2*4, 1, 0],
              [20*2**3, 12*2**2, 6*2, 2, 0, 0],
              [20*4**3, 12*4**2, 6*4, 2, 0, 0]])
b = np.array([10, 60, 0, 0, 0, 0]).T

x = np.linalg.inv(A).dot(b)
a5=x[0]; a4=x[1]; a3=x[2]; a2=x[3]; a1=x[4]; a0=x[5]

print(x)

t = np.linspace(2, 4, 100)
# Posición
q = a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0
# Velocidad
dq = 5*a5*t**4 + 4*a4*t**3 + 3*a3*t**2 + 2*a2*t + a1
# Aceleración
ddq = 20*a5*t**3 + 12*a4*t**2 + 6*a3*t + 2*a2

plt.figure(figsize=(10,10))
plt.subplot(3,1,1); plt.plot(t,q); plt.grid(); plt.title('posición'); plt.ylabel("q [°]")
plt.subplot(3,1,2); plt.plot(t,dq); plt.grid(); plt.title('velocidad'); plt.ylabel("dq [°/s]")
plt.subplot(3,1,3); plt.plot(t,ddq); plt.grid(); plt.title('aceleración'); plt.ylabel("ddq [°/$s^2$]");
plt.show() """

# ------------------------------------------------------
#              Velocidad trapezoidal
# ------------------------------------------------------
def vel_trapezoidal(q0, qf, dqmax, tf, tt):
    q   = np.zeros(tt.shape)
    dq  = np.zeros(tt.shape)
    ddq = np.zeros(tt.shape)

    # Blend time
    tb = (q0-qf+dqmax*tf)/dqmax;

    for i in range(len(tt)):
        t = tt[i]
        if(t <= tb):
            q[i] = q0 + 1./2.*dqmax/tb*t**2
            dq[i] = dqmax/tb*t
            ddq[i] = dqmax/tb
        elif(t <= tf-tb):
            q[i] = q0 - 1./2.*tb*dqmax+dqmax*t
            dq[i] = dqmax
            ddq[i] = 0.0
        else:
            q[i] = qf - 1./2.*dqmax*(t-tf)**2/tb
            dq[i] = -dqmax/tb*(t-tf)
            ddq[i] = -dqmax/tb
    return q, dq, ddq


""" q0 = 10; qf = 60; dqmax = 20; tf = 4;

t = np.linspace(0,4,200)
q, dq, ddq = vel_trapezoidal(q0, qf, dqmax, tf, t)
plt.figure(figsize=(10,10))
plt.subplot(3,1,1); plt.plot(t,q); plt.grid(); plt.title('posición'); plt.ylabel("q [°]")
plt.subplot(3,1,2); plt.plot(t,dq); plt.grid(); plt.title('velocidad'); plt.ylabel("dq [°/s]")
plt.subplot(3,1,3); plt.plot(t,ddq); plt.grid(); plt.title('aceleración'); plt.ylabel("ddq [°/$s^2$]");
plt.show() """

# ------------------------------------------------------
#                     Tiempo minimo
# ------------------------------------------------------
def vel_minimo(q0, qf, ddqmax, tt):
    q   = np.zeros(tt.shape)
    dq  = np.zeros(tt.shape)
    ddq = np.zeros(tt.shape)

    tf = 2*np.sqrt((qf-q0)/ddqmax)
    tb = tf/2;  
    dqmax = ddqmax*tb

    for i in range(len(tt)):
        t = tt[i]
        if(t <= tb):
            q[i] = q0 + 1./2.*dqmax/tb*t**2
            dq[i] = dqmax/tb*t
            ddq[i] = dqmax/tb
        elif(t <= tf-tb):
            q[i] = q0 - 1./2.*tb*dqmax+dqmax*t
            dq[i] = dqmax
            ddq[i] = 0.0
        else:
            q[i] = qf - 1./2.*dqmax*(t-tf)**2/tb
            dq[i] = -dqmax/tb*(t-tf)
            ddq[i] = -dqmax/tb
    return q, dq, ddq

""" q0 = 10; qf = 60; ddqmax = 13.33;

t = np.linspace(0,4,200)
q, dq, ddq = vel_minimo(q0, qf, ddqmax, t)

plt.figure(figsize=(10,10))
plt.subplot(3,1,1); plt.plot(t,q); plt.grid(); plt.title('posición'); plt.ylabel("q [°]")
plt.subplot(3,1,2); plt.plot(t,dq); plt.grid(); plt.title('velocidad'); plt.ylabel("dq [°/s]")
plt.subplot(3,1,3); plt.plot(t,ddq); plt.grid(); plt.title('aceleración'); plt.ylabel("ddq [°/$s^2$]");
plt.show() """