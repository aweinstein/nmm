# -*- coding: utf-8 -*-

'''
F. Wendling, J.-J. Bellanger, F. Bartolomei, and P. Chauvel, “Relevance of
nonlinear lumped-parameter models in the analysis of depth-EEG epileptic
signals,” Biological cybernetics, vol. 83, no. 4, pp. 367–378, 2000.
'''

import numpy as np
from ode_euler import ode_euler
import matplotlib.pyplot as plt

# Parameters
A = 3.25
B = 22
a = 100
b = 50
C = 150
C1 = C
C2 = 0.8 * C
C3 = 0.25 * C
C4 = 0.25 * C

M = 220  #Input mean
D = 22   #Input variance


def s(v, e0 = 2.5, r = 0.56, v0 = 6):
    #Sigmoid function
    return (2 * e0) / (1 + np.exp(r * (v0 - v)))

def f(y, t):
    y0, y1, y2, y3, y4, y5 = y
    
    p = 220 + 22 * np.random.randn()
    y0_dot = y3
    y3_dot = A * a * s(y1 - y2) - 2 * a *y3 - a**2 * y0
    y1_dot = y4
    y4_dot = A * a * (p + C2 * s(C1 * y0)) - 2 *a * y4 - a**2 * y1
    y2_dot = y5
    y5_dot = B * b * (C4 * s(C3 * y0)) - 2 * b * y5 - b**2 * y2
    return np.array((y0_dot, y1_dot, y2_dot, y3_dot, y4_dot, y5_dot))

def plot_sigmoid():
    v = np.linspace(-1, 10)
    plt.plot(v, s(v))
    plt.show()

if __name__ == '__main__':
    np.random.seed(1234)
#    t = np.linspace(0, 1, 10000)
    ic = np.array([  1.62500000e-01,   2.14500000e+01,   3.21991056e+00,
         4.62014664e-16,  -7.97724519e-13,  -8.62602425e-14])
#    ic = 6 * (0,)
#    sol = odeint(f, ic, t)
         
    sol, t = ode_euler(f, ic, 5, 1e-3)
    v1 = sol[:,3]
    v2 = sol[:,4]
    y = v1 - v2
    plt.figure(1)
    plt.clf()
    plt.plot(t, y)
    
    plt.figure(2)    
    plt.clf()
    for i in range(6):
        plt.subplot(7,1,i+1)
        plt.plot(sol[:,i])
    plt.subplot(7,1,7)
#    plt.plot(P)
    
    plt.show()
    
    
    
