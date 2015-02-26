'''Simulation of a Neural Mass Model.

Based on "A neural mass model for MEG/EEG", David and Friston.
'''
import sys
import numpy as np
from scipy.integrate import odeint, ode
from ode_euler import ode_euler
import matplotlib.pyplot as plt

# Parameters from table 1
He = 3.25 # mV
Hi = 22 # mV
tau_e = 10 # ms
tau_i = 20 # ms


def sigmoid(v, k): # ck1, ck2, v0, e0, r):
    '''Sigmoid function.

    The sigmoid function transforms the average membrane potential of the
    population into an average rate of action potentials fired by the neurons.

    The value of the parameters are obtained from table 1 of the paper.

    Parameters
    ----------
    v : array_like
        Membrane potential.
    k : {1, 2, 3}
        Index of the sigmoid.
    Returns
    -------
    float
        Value of the kth-sigmoid evaluated at `v`.
    '''
    v0 = 6 # mV 
    e0 = 5 # 1/s
    r = 0.56 # 1 / mV
    c = 135
    ck = [(c, 0.8 * c),
          (0.25 * c, 0.25 * c),
          (1, 1)]
    try:
        ck1, ck2 = ck[k - 1]
    except IndexError:
        print('Error: `k` must be in (1, 2, 3)')
        sys.exit(1)
    
    S = (ck1 * e0) / (1 + np.exp(r * (v0 - ck2 * v)))
    return S

# odeint likes f(y, t)
def f(y, t):
    x1, x2, x3, v1, v2, v3 = y
    p = 220 + 22 * np.random.randn()
    x1_dot = (He / tau_e) * (p + sigmoid(v2, 1)) - (2 / tau_e) * x1 \
             - (1 / tau_e**2) * v1
    x2_dot = (Hi / tau_i) * sigmoid(v3, 2) - (2 / tau_i) * x2 \
             - (1 / tau_i**2) * v2
    x3_dot = (He / tau_e) * sigmoid(v1 - v2, 3) - (2 / tau_e) * x3 \
             - (1 / tau_e**2) * v3
    v1_dot = x1
    v2_dot = x2
    v3_dot = x3
    return np.array((x1_dot, x2_dot, x3_dot, v1_dot, v2_dot, v3_dot))

# ode likes f(t,y)
def f_ode(t, y):
    x1, x2, x3, v1, v2, v3 = y
    p = 220 + 22 * np.random.randn()
    x1_dot = (He / tau_e) * (p + sigmoid(v2, 1)) - (2 / tau_e) * x1 \
             - (1 / tau_e**2) * v1
    x2_dot = (Hi / tau_i) * sigmoid(v3, 2) - (2 / tau_i) * x2 \
             - (1 / tau_i**2) * v2
    x3_dot = (He / tau_e) * sigmoid(v1 - v2, 3) - (2 / tau_e) * x3 \
             - (1 / tau_e**2) * v3
    v1_dot = x1
    v2_dot = x2
    v3_dot = x3
    return [x1_dot, x2_dot, x3_dot, v1_dot, v2_dot, v3_dot]

def jansen_odeint(tf):
    '''Jansen NMM. '''
    t = np.linspace(0, tf, 100)
    ic = np.random.randn(6) 
    ic = (0,0,0,0,0,0)
    sol = odeint(f, ic, t)
    v1 = sol[:,3]
    v2 = sol[:,4]
    return t, v1 - v2

def jansen_ode(tf):
    '''Jansen NMM. '''
    print('Using ode...')
    y0 = [0,0,0,0,0,0]
    t0 = 0
    r = ode(f_ode).set_integrator('dopri5', nsteps=10000)
    r.set_initial_value(y0, t0)
    dt = 0.01
    while r.successful() and r.t < tf:
        r.integrate(r.t+dt)
        print(r.t, r.y)
    print(r.successful())

def plot_sigmoid():
    v = np.linspace(-0.2, 0.2, 100)
    s1 = sigmoid(v, 1)
    s2 = sigmoid(v, 2)
    s3 = sigmoid(v, 3)
    plt.figure()
    plt.plot(v, s1, v, s2, v, s3)
    plt.show()
    print('Done')

if __name__ == '__main__':
    np.random.seed(1234)
    ic = 6 * (0,)
    sol, t = ode_euler(f, ic, 1, 1e-3)
    v1 = sol[:,3]
    v2 = sol[:,4]
    y = v1 - v2
    plt.plot(t, y)
    plt.show()


# if __name__ == '__main__':
#     np.random.seed(1234)
#     jansen_ode(10)
    
# if __name__ == '__main__x':
#     t, y = jansen(1)
#     plt.close('all')
#     plt.plot(t, y)
#     plt.show()
#     print('Done')
    


   
