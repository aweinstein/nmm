'''
F. Wendling, J.-J. Bellanger, F. Bartolomei, and P. Chauvel, “Relevance of
nonlinear lumped-parameter models in the analysis of depth-EEG epileptic
signals,” Biological cybernetics, vol. 83, no. 4, pp. 367–378, 2000.
'''

import numpy as np
from scipy.integrate import odeint
from ode_euler import ode_euler
import matplotlib.pyplot as plt

def s(v):
    '''Sigmoid function.'''
    e0 = 2.5
    r = 6
    v0 = 6
    return (2 * e0) / (1 + np.exp(r * (v0 - v)))

def f(y, t):
    y0, y1, y2, y3, y4, y5 = y
    #p = 220 + 22 * np.random.randn()
    p = 120 + 200 * np.random.rand()
    A = 3.25
    B = 22
    a = 100
    b = 50
    C = 135
    C1 = C
    C2 = 0.8 * C
    C3 = 0.25 * C
    C4 = 0.25 * C

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
    t = np.linspace(0, 0.51, 10000)
    ic = 6 * (0,)
    #sol = odeint(f, ic, t)
    sol, t = ode_euler(f, ic, 1, 1e-3)
    v1 = sol[:,3]
    v2 = sol[:,4]
    y = v1 - v2
    plt.plot(t, y)
    plt.show()
