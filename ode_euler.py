import numpy as np

def ode_euler(f, y0, tf, h):
    '''Solve and ODE using Euler method.
    Solve the ODE y_dot = f(y, t)
    Parameters
    ----------
    f : function
        Function describing the ODE
    y0 : array_like
        Initial conditions.
    tf : float
        Final time.
    h : float
        Time step
    Returns
    -------
    y : array_like
        Solution to the ODE.
    t : array_like
        Time vector.
    '''
    y0 = np.array(y0)
    ts = np.arange(0, tf+h, h)
    y = np.empty((ts.size, y0.size))
    y[0,:] = y0
    for t, i in zip(ts[1:], range(ts.size - 1)):
        y[i+1,:] = y[i,:] + h * f(y[i,:], t)
    return y, ts
