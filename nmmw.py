'''
F. Wendling, J.-J. Bellanger, F. Bartolomei, and P. Chauvel, "Relevance of
nonlinear lumped-parameter models in the analysis of depth-EEG epileptic
signals," Biological cybernetics, vol. 83, no. 4, pp. 367-378, 2000.
'''
from __future__ import division

import numpy as np
#from scipy.integrate import odeint
from ode_euler import ode_euler
import matplotlib.pyplot as plt

def s(v):
    '''Sigmoid function.'''
    e0 = 2.5
    r = 0.56
    v0 = 6
    return (2 * e0) / (1 + np.exp(r * (v0 - v)))

def f(y, t, C=68):
    y0, y1, y2, y3, y4, y5 = y
    #p = 220 + 22 * np.random.randn() 
    #Page 4 of Jansen et al, "p(t) will have
    # an amplitude varying between 120 and 320 pulses per second."
    p = 120 + 200 * np.random.rand() 
    A = 3.25
    B = 22
    a = 100
    b = 50
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
    Cs = (68, 128, 135, 270, 675, 1350)
    EEGs = []
    for C in Cs:
        # Run it once for a brief time to get sensible initial conditions
        ic = 6 * (0,)
        sol, t = ode_euler(f, ic, 0.3, 1e-3, params=(C,))
        ic = sol[-1,:]
        # Run it again with the new initial conditions
        sol, t = ode_euler(f, ic, 2, 1e-3, params=(C,))

        y0, y1, y2 = sol[:,:3].T
        EEGs.append(y1 - y2)


    # f, axs = plt.subplots(6, figsize=(21,8))
    # for ax, var in zip(axs, sol.T):
    #     ax.plot(t, var)
    #     ax.set_xticks([])
    # plt.tight_layout()

    # plt.figure(figsize=(21,3))
    # plt.plot(t, y1 - y2)

    plt.close('all')
    f, axs = plt.subplots(len(EEGs), figsize=(21,10))
    for EEG, ax, C in zip(EEGs, axs, Cs):
        ax.plot(t, EEG)
        #ax.set_xticks([])
        ax.set_title('C = %d' % C)
    ax.set_xlabel('time [s]')
    plt.tight_layout()
    plt.savefig('fig_3_jansen_rit.pdf')
    plt.show()

        

