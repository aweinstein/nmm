import numpy as np

def ode_eulerM(f, y0, tf, h, params=()):
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
    params : tuple, optional
        Optional list of parameters to be passed to `f`.
    Returns
    -------
    y : array_like
        Solution to the ODE.
    t : array_like
        Time vector.
    '''
    ts = np.arange(0, tf+h, h)

    y = np.zeros((ts.size, y0.shape[0], y0.shape[1]))
    y[0,:,:] = y0
    for t, i in zip(ts[1:], range(ts.size - 1)):
        y[i+1,:,:] = y[i,:,:] + h * f(y[i,:,:], t, *params)
    return y, ts
