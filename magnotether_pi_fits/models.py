import numpy as np

def p_control_model(t, d, kp, xs, x0):
    """ Exact solution for proportional control model given a step input in xs.

    Parameters
    ----------
    t : ndarray
        array of time points
    d : float
        damping coefficient
    kp : float
        proportional gain
    xs : float
        set-point value for step input
    x0: float
        initial condition for system state

    Returns
    -------
    x : ndarray
        array of state values, x, corresponding to time points in array t

    """
    d = np.absolute(d)
    kp = np.absolute(kp)
    x_eq = kp*xs/(d + kp)
    x = x_eq + (x0 - x_eq)*np.exp(-(d+kp)*t)
    return x


def p_control_derivs(t, d, kp, xs, x0):
    """ Returns derivative dx/dt response of proportional control model to a
    step response input. 

    Parameters
    ---------
    t : ndarray
        array of time points
    d : float
        damping coefficient
    kp : float
        proportional gain
    xs : float
        set-point value for step input
    x0: float
        initial condition for system state
    
    """
    x = p_control_model(t, d, kp, xs, x0)
    dx_dt = -d*x + kp*(xs - x)
    return dx_dt


def p_control_cost(param, xs, t, x):
    """ Cost function for proportional controller model. For fitting
    proportional controller to step input.  

    Parameters
    ----------
    param : ndarray
        array of fit parameters [d, kp] 
    xs : float 
        set-point value
    t : ndarray
        array of time values 
    x : ndarray
        array of state values to be fit by model 

    Returns
    -------
    cost : float
        sum of the squared errors for the fit

    """
    d, kp = param
    x_eq  = kp*xs/(d + kp)
    x_fit = p_control_model(t, d, kp, xs, x[0])
    cost = ((x_fit - x)**2).sum()
    return cost


def pi_control_model(t, d, kp, ki, xs, x0):
    """ Exact solution for proportional integral control model given a step
    input in 0.

    Parameters
    ----------
    t : ndarray
        array of time points
    d : float
        damping coefficient
    kp : float
        proportional gain
    xs : float
        set-point value for step input
    x0: float
        initial condition for system state

    Returns
    -------
    x : ndarray
        array of state values, x,  corresponding to time points in array t

    """
    d = np.absolute(d)
    kp = np.absolute(kp)
    ki = np.absolute(ki)

    disc = (d+kp)**2 - 4*ki + 0j
    eig0 = 0.5*(-(d+kp) + np.sqrt(disc))
    eig1 = 0.5*(-(d+kp) - np.sqrt(disc))
    dx0 = -d*x0 + kp*(xs - x0) 
    c0 = (eig1*(xs - x0) + dx0 )/(eig0 - eig1)
    c1 = x0 - xs - c0
    x = np.real(xs + c0*np.exp(eig0*t) + c1*np.exp(eig1*t))
    return x 


def pi_control_derivs(t, d, kp, ki, xs, x0, n=1):
    """ Returns derivative dx/dt (or d^2x/dt^2) response of proportional
    integral control model to a step response input. 

    Parameters
    ----------
    t : ndarray
        array of time points
    d : float
        damping coefficient
    kp : float
        proportional gain
    xs : float
        set-point value for step input
    x0: float
        initial condition for system state

    Returns
    -------
    dx_dt : ndarray
        array of derivatives of state variables, dx_dt, corresponding to the time
        points in the array t.

    """

    dt = t[1] - t[0]
    x = pi_control_model(t, d, kp, ki, xs, x0)
    eta = xs - x
    dx_dt = -d*x + kp*(xs - x) + ki*eta
    d2x_dt2 = -(d + kp)*dx_dt + ki*(xs - x)
    match n:
        case 0:
            ret_val = x
        case 1:
            ret_val = dx_dt
        case 2:
            ret_val = d2x_dt2
    return ret_val


def pi_control_cost(param, xs, t, x, imask=None):
    """ Cost function for proportional integral controller model. For fitting
    proportional integral controller to step input.  

    Parameters
    ----------
    param : ndarray
        array of fit parameters [d, kp, ki] 
    xs : float 
        set-point value
    t : ndarray
        array of time values 
    x : ndarray
        array of state values to be fit by model 

    Returns
    -------
    cost : float
        sum of the squared errors for the fit
    """
    d, kp, ki = param
    x_fit = pi_control_model(t, d, kp, ki, xs, x[0])
    resid = (x_fit - x)**2
    if np.any(imask):
        resid = resid[~imask]
    cost = resid.sum()
    return cost


def pi_w_openloop_control_cost(param, xs, t_cl, x_cl, t_ol, x_ol, imask_cl=None, imask_ol=None):
    """ Cost function for proportional integral controller model. For fitting
    proportional integral controller to step input followed by a section of
    data with where system is in openloop.  

    Parameters
    ----------
    param : ndarray
        array of fit parameters [d, kp, ki]
    xs : float
        set-point value
    t_cl : ndarray
        1D array of time values for closed-loop data section (float)
    x_cl : ndarray
        1D array of state values for closed-loop data section (float)
    t_ol : ndarray
        1D array of time values for open-loop data section (float)
    x_ol : ndarray
        1D array of state values for open-loop data section (float)
    imask_cl : ndarray (optional)
        1D mask array specifying which values in closed-loop data to ignore 
        when performing fit (bool)
    imask_ol : ndarray (optional)
        1d mask array specifying which values in open-loop data to ignore
        when performing fit (bool)

    Returns:
        cost : float
        sum of squared errors for fit 
    """

    d, kp, ki = np.absolute(param)

    # Get cost for closed-loop section
    x_cl_fit = pi_control_model(t_cl, d, kp, ki, xs, x_cl[0])
    resid_cl = (x_cl_fit - x_cl)**2
    if np.any(imask_cl):
        resid_cl = resid_cl[~imask_cl]
    cost_cl = resid_cl.sum()

    # Get equilibrium solution for open-loop section 
    ierr = np.trapz(xs - x_cl_fit, dx=t_cl[1]-t_cl[0]) 
    xeq = (ki*ierr)/d

    # Get cost for open-loop sections
    x_ol_fit = exponential_decay(t_ol, d, x_cl_fit[-1], xeq)
    resid_ol = (x_ol_fit - x_ol)**2
    if np.any(imask_ol):
        resid_ol = resid_ol[~imask_ol]
    cost_ol = resid_ol.sum()

    n_cl = len(resid_cl)
    n_ol = len(resid_ol)
    cost = cost_cl/n_cl + cost_ol/n_ol
    return cost


def pi_ensemble_cost(param, fit_data_dict):

    """ Cost function for ensemble fit of proportional integral controller
    model.  For simultaneuous fitting to dataset the response of a proportional
    integral controller to step input followed by a section of data with where
    system is in openloop.  

    Parameters
    ----------
    param : ndarray
        array of fit parameters [d, kp, ki]
    fit_data_dict : dict
        dictionary of fit_data for each duration

    Returns:
        cost : float
        sum of squared errors for fit 
    """

    total_cost = 0.0
    for duration, fit_data in fit_data_dict.items():
        model_args = fit_data['model_args']
        model_cost = fit_data['model_cost']
        cost = model_cost(param, *model_args)
        total_cost += cost
    return total_cost


def constrained_pi_control_cost(param, xs, x0_ol, x1_ol, dt_ol, ierr, t, x):
    """ Cost function for constrained proportional integral controller model.
    For fitting proportional integral controller to a step input where the
    values during subsequent openloop period are specified. 

    Parameters
    ----------
    param : ndarray
        array of fit parameters [d, kp] 
    xs : float 
        set-point value
    x0_ol : float
        value at start of open loop period
    x1_ol : float
        value at end of open loop period
    dt_ol : float
        duration of open loop period
    ierr : float
        integral gain during close loop period
    t : ndarray
        array of time values 
    x : ndarray
        array of state values to be fit by model 

    Returns
    -------
    cost : float
        sum of the squared errors for the fit
    """
    d, kp = param
    ki = ki_for_constrained_pi(d, x0_ol, x1_ol, dt_ol, ierr)
    x_fit = pi_control_model(t, d, kp, ki, xs, x[0])
    cost = ((x_fit - x)**2).sum()
    return cost


def ki_for_constrained_pi(d, x0_ol, x1_ol, dt_ol, ierr):
    """
    Returns the ki for the constrained proportional integral model. The
    Constraint ensures that during the open-loop period the integral gain is
    such that the system goes through the points x0_ol at t0 and x1_ol at t1 =
    t0 + dt_ol given the damping, d, and the integral of the error, ierr, at
    the end of the closed loop period. 

    Parameters
    ----------
    d : float
        damping coefficient
    x0_ol : float
        system state at start of the open-loop period. 
    x1_ol : float
        system state at the end of hte open-loop period
    dt_ol : float
        the duration of the open-loop period
    ierr : float
        the integral of the error at the end of the closed-loop period. 

    Returns
    -------
    ki : float
        integral gain
    """
    xeq_ol = equilib_value(0, dt_ol, x0_ol, x1_ol, d)
    ki = (xeq_ol*d)/ierr
    return ki


def exponential_decay(t, d, x0, xeq):
    """ Returns exponential decay, x(t), where  

    x(t) = xeq + (x0 - xeq)*np.exp(-d*(t-t[0]))

    Parameters
    ----------
    t : ndarray 
        array of time points at which to evaluate x(t)
    d : float
        dampling coefficient
    x0 : float
        initial state value at t = t[0]
    xeq : float
        the equilibruim value for system

    Returns
    -------
    x : ndarray
        array of state values at time points in t

    """
    x = xeq + (x0 - xeq)*np.exp(-d*(t-t[0]))
    return x


def equilib_value(t0, t1, x0, x1, a):
    """ Returns equilibrium value, xeq,  for a first order system

    dx/dt = -a*x + b

    where values for x0 and x1 for x are given at time points t0 and t1
    and the constant a in known. 

    Parameters
    ---------
    t0 : float
        1st time value
    t1 : float
        2nd time value
    x0 : float
        state value at time t0
    x1 : float
        state value at time t1
    a : float
        known system parameter

    Returns
    -------

    xeq : float
        the equilibrium value for the linear system

    """

    dt = t1 - t0
    beta = np.exp(-dt*a)
    xeq = (x1 - x0*beta)/(1 - beta)
    return xeq





