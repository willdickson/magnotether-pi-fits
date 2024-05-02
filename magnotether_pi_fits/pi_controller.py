import copy
import collections
import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from . import utility
from . import models


def ensemble_fit_pi_controller(data_dir, guess=None, omega_min=20.0, tfrac=None, include_ol=False, 
        bounds=None, fit_win_cl=None, fit_win_ol=None,  disp=False):
    """ Fits proportional integral controller to all magnotether datasets in given data 
    directory in an ensemble fashion, i.e. trys to fit a single model to all data sets
    where every dataset is given equal weighting. 

    Parameters
    ----------
    data_dir : Path or str
        directory containing datasets
    guess : array_like
        initial guess for controller parameters [d, kp, ki]
    omega_min : float  (optional)
        minimum required omega or None. Used as starting point for fit. Start
        of stimulus is used if omega_min is None. 
    tfrac : float or None
        fraction of stimulus duration to fit (all if None)
    include_ol : bool (optional) 
        flag whether to include or not include open loop data section
    bounds: tuple or None (optional)
        bounds for the fit parameters. Tuple of tuples e.g. 
        bounds = (
            (d lower bound,   d lower bound), 
            (kp lower bound   kp upper bound),
            (ki lower bounds, ki upper bound),
            )
    fit_win_cl : float or None (optional)
        time window from start of step used for fitting closed loop-period
    fit_win_ol : float or None (optional)
        time window from end of open-loop period used for fitting open-loop
        period
    disp : bool
        flag specifying whether or not to display results as they are generated

    Returns
    -------
    results_dict : dict
        dictionary of fit results (and original data) for each dataset
        results_dict= {
                'd'        : d, 
                'kp'       : kp, 
                'ki'       : ki,
                'duration' : {
                    duration1 : {
                        'ierr'     : {
                            'model' : ierr_model, 
                            'data'  : ierr_data, 
                        }, 
                        'closed_loop' : {
                            't'        : t_cl, 
                            'omega'    : omega_cl, 
                            'setpt'    : setpt_cl,
                            'disable'  : disable_cl,
                            'omega_fit': omega_fit_cl,
                            },
                        'open_loop': {
                            't'        : t_ol, 
                            'omega'    : omega_ol, 
                            'setpt'    : setpt_ol,
                            'disable'  : disable_ol,
                            'omega_fit': omega_fit_ol,
                            'omega_eq' : omega_eq_ol,
                            }
                    duration2 : { 
                        ...
                        },
                    ...
                    }
                }
    """
    if type(data_dir) == str:
        data_dir = pathlib.Path(data_dir)
    raw_data_dict = utility.get_data_sorted_by_duration(data_dir)
    fit_data_dict = collections.OrderedDict()
    if disp:
        print()
        print('loading data')
    for duration, raw_data in raw_data_dict.items():
        if disp:
            print(f'  duration = {duration}')
        fit_data = extract_data_for_pi_fit(
                raw_data, 
                omega_min, 
                tfrac, 
                include_ol, 
                bounds, 
                fit_win_cl, 
                fit_win_ol
                ) 
        fit_data_dict[duration] = fit_data
    if disp:
        print()
    if guess is None:
        guess = [0.0, 1.0, 0.0]
    res = optimize.minimize(models.pi_ensemble_cost, guess, (fit_data_dict,), bounds=bounds)
    d, kp, ki = np.absolute(res.x)

    if disp:
        print('ensemble fit parameters')
        print(f'  d  = {d}')
        print(f'  kp = {kp}')
        print(f'  ki = {ki}')
        print()

    # Create results data dictionary
    results_dict = {
            'd'       : d, 
            'kp'      : kp, 
            'ki'      : ki, 
            'type'    : 'ensemble', 
            'duration': {},
            }
    for duration, fit_data in fit_data_dict.items(): 
        data_cl = fit_data['closed_loop']
        data_ol = fit_data['open_loop']
        ierr_data = fit_data['ierr']

        # Copy data for closed and open loop periods
        trial_data = {}
        trial_data['closed_loop'] = copy.deepcopy(data_cl)
        trial_data['open_loop'] = copy.deepcopy(data_ol)

        # Get prediction during closed-loop section
        t_cl = data_cl['t']
        setpt_cl = data_cl['setpt']
        omega_cl = data_cl['omega']
        omega_fit_cl = models.pi_control_model(t_cl, d, kp, ki, np.median(setpt_cl), omega_cl[0]) 
        trial_data['closed_loop']['omega_fit'] = omega_fit_cl

        # Get prediction during open loop period
        t_ol = data_ol['t']
        omega_ol = data_ol['omega']
        setpt_ol = data_ol['setpt']
        omega_init_ol = omega_fit_cl[-1]
        ierr_model = np.trapz(setpt_cl - omega_fit_cl, dx=t_ol[1] - t_ol[0])
        omega_eq_ol = ki*ierr_model/d
        omega_fit_ol = models.exponential_decay(t_ol, d, omega_init_ol, omega_eq_ol)
        trial_data['open_loop']['omega_eq'] = omega_eq_ol
        trial_data['open_loop']['omega_init'] = omega_init_ol
        trial_data['open_loop']['omega_fit'] = omega_fit_ol
        trial_data['ierr'] = {
                'model' : ierr_model, 
                'data'  : ierr_data,
                }
        results_dict['duration'][duration] = trial_data
    return results_dict


def fit_pi_controller_to_datasets(data_dir, guess=None, omega_min=20.0, tfrac=None, include_ol=False, 
        bounds=None, fit_win_cl=None, fit_win_ol=None, disp=False, ):
    """ Fits proportional integral controller individually to all magnotether
    datasets in given data directory.

    Parameters
    ----------
    data_dir : Path or str
        directory containing datasets
    guess : array_like
        initial guess for controller parameters [d, kp, ki]
    omega_min : float  (optional)
        minimum required omega or None. Used as starting point for fit. Start
        of stimulus is used if omega_min is None. 
    tfrac : float or None
        fraction of stimulus duration to fit (all if None)
    include_ol : bool (optional) 
        flag whether to include or not include open loop data section
    bounds: tuple or None (optional)
        bounds for the fit parameters. Tuple of tuples e.g. 
        bounds = (
            (d lower bound,   d lower bound), 
            (kp lower bound   kp upper bound),
            (ki lower bounds, ki upper bound),
            )
    fit_win_cl : float or None (optional)
        time window from start of step used for fitting closed loop-period
    fit_win_ol : float or None (optional)
        time window from end of open-loop period used for fitting open-loop
        period
    disp : bool
        flag specifying whether or not to display results as they are generated

    Returns
    -------
    results_dict : dict
        dictionary of fit results (and original data) for each dataset
        results_dict = {
            'type'     : 'individual',
            'duration' : {
                duration1 : {
                    'd'        : d, 
                    'kp'       : kp, 
                    'ki'       : ki,
                    'ierr'     : {
                        'model' : ierr_model, 
                        'data'  : ierr_data, 
                    }, 
                    'closed_loop' : {
                        't'        : t_cl, 
                        'omega'    : omega_cl, 
                        'setpt'    : setpt_cl,
                        'disable'  : disable_cl,
                        'omega_fit': omega_fit_cl,
                        },
                    'open_loop': {
                        't'        : t_ol, 
                        'omega'    : omega_ol, 
                        'setpt'    : setpt_ol,
                        'disable'  : disable_ol,
                        'omega_fit': omega_fit_ol,
                        'omega_eq' : omega_eq_ol,
                        }
                    }
                'duration2' : {
                    ...
                    }
                ...
                }
            }
    """

    if disp:
        print()
    if type(data_dir) == str:
        data_dir = pathlib.Path(data_dir)
    data_dict = utility.get_data_sorted_by_duration(data_dir)
    max_duration = max([d for d,_ in data_dict.items()])
    results_dict = {'type': 'individual', 'duration' : {}}
    for duration, data in data_dict.items():
        if disp:
            print(f'duration = {duration}')
        result = fit_pi_controller(
                data, 
                guess=guess, 
                omega_min=omega_min, 
                tfrac=tfrac, 
                include_ol=include_ol, 
                bounds=bounds,
                fit_win_cl=fit_win_cl,
                fit_win_ol=fit_win_ol,
                )
        if disp:
            print(f'  d       = {result["d"]}')
            print(f'  kp      = {result["kp"]}')
            print(f'  ki      = {result["ki"]}')
            print()
            data_cl = result['closed_loop']
            data_ol = result['open_loop']

            # Extract closed loop data
            t_cl = data_cl['t']
            setpt_cl = data_cl['setpt']
            omega_cl = data_cl['omega']
            omega_fit_cl = data_cl['omega_fit']

            # Extract open loop data
            t_ol = data_ol['t']
            setpt_ol = data_ol['setpt']
            omega_ol = data_ol['omega']
            omega_fit_ol = data_ol['omega_fit']

            results_dict['duration'][duration] = {
                    'd'           : result['d'], 
                    'kp'          : result['kp'], 
                    'ki'          : result['ki'],
                    'ierr'        : copy.deepcopy(result['ierr']),
                    'closed_loop' : copy.deepcopy(data_cl), 
                    'open_loop'   : copy.deepcopy(data_ol),
                    }

    return results_dict


def fit_pi_controller(data, guess=None, omega_min=20.0, tfrac=None, include_ol=False, 
        bounds=None, fit_win_cl=None, fit_win_ol=None):
    """ Fits proportional integral controller to all magnotether data. Currently 
    only uses closed loop data section. 

    Parameters
    ----------
    data_dir : Path or str
        directory containing datasets
    guess : array_like
        initial guess for controller parameters [d, kp, ki]
    omega_min : float  (optional)
        minimum required omega or None. Used as starting point for fit. Start
        of stimulus is used if omega_min is None. 
    tfrac : float or None (optional)
        fraction of stimulus duration to fit (all if None)
    include_ol : bool (optional) 
        flag whether to include or not include open loop data section
    bounds: tuple or None (optional)
        bounds for the fit parameters. Tuple of tuples e.g. 
        bounds = (
            (d lower bound,   d lower bound), 
            (kp lower bound   kp upper bound),
            (ki lower bounds, ki upper bound),
            )
    fit_win_cl : float or None (optional)
        time window from start of step used for fitting closed loop-period
    fit_win_ol : float or None (optional)
        time window from end of open-loop period used for fitting open-loop
        period
        

    Returns
    -------
    result : dict
        dictionary containing the resuls of the fit
        result = {
                'd'        : d, 
                'kp'       : kp, 
                'ki'       : ki, 
                'ierr'     : {
                    'model' : ierr_model, 
                    'data'  : ierr_data, 
                    }, 
                'closed_loop' : {
                    't'        : t_cl, 
                    'omega'    : omega_cl, 
                    'setpt'    : setpt_cl,
                    'disable'  : disable_cl,
                    'omega_fit': omega_fit_cl,
                    },
                'open_loop': {
                    't'        : t_ol, 
                    'omega'    : omega_ol, 
                    'setpt'    : setpt_ol,
                    'disable'  : disable_ol,
                    'omega_fit': omega_fit_ol,
                    'omega_eq' : xeq_ol,
                    }
                }

    """
    if guess is None:
        guess = [0.0, 1.0, 0.0]

    fit_data = extract_data_for_pi_fit(
            data, 
            omega_min, 
            tfrac, 
            include_ol, 
            bounds, 
            fit_win_cl, 
            fit_win_ol
            ) 

    data_cl = fit_data['closed_loop']
    data_ol = fit_data['open_loop']

    # Fit model
    model_cost = fit_data['model_cost']
    model_args = fit_data['model_args']
    res = optimize.minimize(model_cost, guess, model_args, bounds=bounds)
    d, kp, ki = np.absolute(res.x)

    # Extract data arrays for close loop period
    t_cl = data_cl['t']
    omega_cl = data_cl['omega']
    setpt_cl = data_cl['setpt']
    disable_cl = data_cl['disable']

    # Extract data arrays for open loop period
    t_ol = data_ol['t']
    omega_ol = data_ol['omega']
    setpt_ol = data_ol['setpt']
    disable_ol = data_ol['disable']

    # Get prediction during closed-loop section
    omega_fit_cl = models.pi_control_model(t_cl, d, kp, ki, setpt_cl[0], omega_cl[0])

    ierr_data = fit_data['ierr']
    ierr_model = np.trapz(setpt_cl[0] - omega_fit_cl, dx=t_cl[1]-t_cl[0]) 
    t0_ol = data_ol['t'][0]
    t1_ol = data_ol['t'][-1]
    x0_ol = omega_fit_cl[-1]
    xeq_ol = ki*ierr_model/d
    omega_fit_ol = models.exponential_decay(t_ol, d, x0_ol, xeq_ol)

    result = {
            'd'        : d, 
            'kp'       : kp, 
            'ki'       : ki, 
            'ierr'     : {
                'model' : ierr_model, 
                'data'  : ierr_data, 
                }, 
            'closed_loop' : {
                't'        : t_cl, 
                'omega'    : omega_cl, 
                'setpt'    : setpt_cl,
                'disable'  : disable_cl,
                'omega_fit': omega_fit_cl,
                'imask'    : data_cl['imask'],
                },
            'open_loop': {
                't'        : t_ol, 
                'omega'    : omega_ol, 
                'setpt'    : setpt_ol,
                'disable'  : disable_ol,
                'omega_fit': omega_fit_ol,
                'omega_eq' : xeq_ol,
                'imask'    : data_cl['imask'],
                }
            }
    return result


def extract_data_for_pi_fit(data, omega_min=20.0, tfrac=None, include_ol=False, 
        bounds=None, fit_win_cl=None, fit_win_ol=None):
    """ Extracts closed-loop and open-loop data sections for controller fit. 

    Parameters
    ----------
    data_dir : Path or str
        directory containing datasets
    omega_min : float  (optional)
        minimum required omega or None. Used as starting point for fit. Start
        of stimulus is used if omega_min is None. 
    tfrac : float or None (optional)
        fraction of stimulus duration to fit (all if None)
    include_ol : bool (optional)
        flag whether to include or not include open loop data section
    bounds: tuple or None (optional)
        bounds for the fit parameters. Tuple of tuples e.g. 
        bounds = (
            (d lower bound,   d lower bound), 
            (kp lower bound   kp upper bound),
            (ki lower bounds, ki upper bound),
            )
    fit_win_cl : float or None (optional)
        time window from start of step used for fitting closed loop-period
    fit_win_ol : float or None (optional)
        time window from end of open-loop period used for fitting open-loop
        period
        bounds=None, fit_win_cl=None, fit_win_ol=None):

    Returns
    ------
    fit_data : dict

        fit_data = {
                'model_args' : model_args, 
                'model_cost' : model_cost, 
                'ierr'       : ierr, 
                'closed_loop' : {
                    't'        : t_cl, 
                    'omega'    : omega_cl, 
                    'setpt'    : setpt_cl,
                    'disable'  : disable_cl,
                    },
                'open_loop': {
                    't'        : t_ol, 
                    'omega'    : omega_ol, 
                    'setpt'    : setpt_ol,
                    'disable'  : disable_ol,
                    }
                }
    """
    data = copy.deepcopy(data)
    data = utility.fix_setpt_error(data)
    mask = data['setpt'] > 0
    t_start = data['t'][mask][0]
    duration = utility.get_duration(data)
    if omega_min is None:
        # Apply mask so that data begins when stimulus starts
        mask = data['t'] >= t_start 
        data = utility.apply_mask(data, mask)
    else:
        # Apply mask so that data begins 1 second before stimulus starts
        mask = data['t'] > t_start - 1.0
        data = utility.apply_mask(data, mask)
        # Apply mask to that data begins when omega first reaches omega_min
        mask = data['omega'] > omega_min
        data = utility.apply_mask(data, mask)
        t_offset = data['t'][0]
        data['t'] = data['t'] - data['t'][0]

    # Get openloop data
    mask = data['t'] >= duration
    data_ol = utility.apply_mask(data, mask)

    # Get closed loop data
    mask = data['disable'] == 0
    data_cl = utility.apply_mask(data, mask)

    #Truncate data to fraction of stimulus duration 
    if tfrac is not None:
        mask = data['t'] < tfrac*duration
        data_cl = utility.apply_mask(data_cl, mask)

    # Extract data arrays for close loop period
    t_cl = data_cl['t']
    omega_cl = data_cl['omega']
    setpt_cl = data_cl['setpt']
    disable_cl = data_cl['disable']

    # Extract data arrays for open loop period
    t_ol = data_ol['t']
    omega_ol = data_ol['omega']
    setpt_ol = data_ol['setpt']
    disable_ol = data_ol['disable']

    # Estimate integral of the error during closed loop period
    dt = data_cl['t'][1] - data_cl['t'][0]
    err = data_cl['setpt'] - omega_cl
    ierr = np.trapz(err, dx=dt)

    # Fit data via least squares using scipy's minimuze. 
    if fit_win_cl:
        imask_cl = (t_cl - t_cl[0]) > fit_win_cl
    else:
        imask_cl = np.zeros_like(t_cl, dtype=np.dtype('bool'))

    if include_ol:
        if fit_win_ol:
            imask_ol = (t_ol - t_ol[0]) < (t_ol[-1] - t_ol[0]) - fit_win_ol 
            print(imask_ol)
        else:
            imask_ol = np.zeros_like(t_ol, dtype=np.dtype('bool'))
        model_args = (setpt_cl[0], t_cl, omega_cl, t_ol, omega_ol, imask_cl, imask_ol)
        model_cost = models.pi_w_openloop_control_cost
    else:
        model_args = (setpt_cl[0], t_cl, omega_cl, imask_cl) 
        model_cost = models.pi_control_cost

    fit_data = {
            'model_args' : model_args, 
            'model_cost' : model_cost, 
            'ierr'       : ierr, 
            'closed_loop' : {
                't'        : t_cl, 
                'omega'    : omega_cl, 
                'setpt'    : setpt_cl,
                'disable'  : disable_cl,
                'imask'    : imask_cl,
                },
            'open_loop': {
                't'        : t_ol, 
                'omega'    : omega_ol, 
                'setpt'    : setpt_ol,
                'disable'  : disable_ol,
                'imask'    : imask_ol,
                }
            }
    return fit_data


def plot_omega_fits(results_dict, figsize=(16,12), save_name=None, title_str=None):
    """  Plots the magnotether data fits for omega vs time.  

    Parameters
    ----------
    results_dict : dict
        dictionary of fit results as returned by either ensemble_fit_pi_controller or
        fit_pi_controller_to_datasets
    figsize : tuple
        size of figure (width, height)
    save_name : str
        name of saved figure file or None
    title_str : str
        title string for figure
        
    """
    plot_cnt = 0
    num_duration = len(results_dict['duration'])
    fig, ax = plt.subplots(num_duration, 1, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0.1)
    for duration, trial_data in results_dict['duration'].items():
        data_cl = trial_data['closed_loop']
        data_ol = trial_data['open_loop']

        # Extract closed loop data
        t_cl = data_cl['t']
        setpt_cl = data_cl['setpt']
        omega_cl = data_cl['omega']
        omega_fit_cl = data_cl['omega_fit']
        imask_cl = data_cl['imask']

        # Extract open loop data
        t_ol = data_ol['t']
        setpt_ol = data_ol['setpt']
        omega_ol = data_ol['omega']
        omega_fit_ol = data_ol['omega_fit']
        imask_ol = data_cl['imask']

        # Plot closed loop data
        ax[plot_cnt].plot(t_cl, omega_cl, 'o', c='gray', alpha=0.4)
        cl_fit_line,  = ax[plot_cnt].plot(t_cl, omega_fit_cl, 'r', linewidth=2)
        setpt_line, = ax[plot_cnt].plot(t_cl, setpt_cl, 'k', linewidth=2)

        # plot open loop data
        ax[plot_cnt].plot(t_ol, omega_ol, 'o', c='gray', alpha=0.4)
        ol_fit_line, = ax[plot_cnt].plot(t_ol, omega_fit_ol, 'b', linewidth=2)

        ax[plot_cnt].set_ylabel(r'$\omega$')
        ax[plot_cnt].grid(True)
        ax[plot_cnt].set_ylim(0, setpt_cl[0]*1.1)
        if plot_cnt == 0:
            if title_str is not None:
                ax[plot_cnt].set_title(title_str)
            ax[plot_cnt].legend(
                    (setpt_line, cl_fit_line, ol_fit_line), 
                    ('set point', 'fit closed-loop', 'fit open-loop' ), 
                    loc='center right',
                    )
        plot_cnt += 1
    ax[plot_cnt-1].set_xlabel('t (sec)')
    if save_name is not None:
        fig.savefig(f'{save_name}')
    plt.show()


def plot_deriv_fits(results_dict, n=1, title_str=None, figsize=(20,12), save_name=None):
    """ Plots derivative of fits of pi control model to magnotether data.

    Parameters
    ----------
    results_dict : dict
        dictionary of fit results as returned by either ensemble_fit_pi_controller or
        fit_pi_controller_to_datasets
    n : int
        order of the derivative to plot n=1 or n=2
    figsize : tuple
        size of figure (width, height)
    save_name : str
        name of saved figure file or None
    title_str : str
        title string for figure
    ---------

    """

    num_duration = len(results_dict['duration'])
    ngrid = int(np.ceil(np.sqrt(num_duration)))
    fig, axs = plt.subplots(ngrid, ngrid, figsize=figsize, sharex=True)
    cnt = 0
    for duration, trial_data in results_dict['duration'].items():
        if results_dict['type'] == 'individual':
            d = trial_data['d']
            ki = trial_data['ki']
            kp = trial_data['kp']
        else:
            d = results_dict['d']
            ki = results_dict['ki']
            kp = results_dict['kp']
        data_cl = trial_data['closed_loop']
        data_ol = trial_data['open_loop']

        t_cl = data_cl['t']
        setpt_cl = data_cl['setpt']
        omega_cl = data_cl['omega']
        omega_fit_cl = data_cl['omega_fit']
        imask_cl = data_cl['imask']
        deriv =  models.pi_control_derivs(t_cl, d, kp, ki, setpt_cl[0], omega_cl[0], n=n)
        match n:
            case 1:
                dt = t_cl[1] - t_cl[0]
                eta = integrate.cumulative_trapezoid(setpt_cl - omega_cl, dx=dt)
                y = deriv - ki*(setpt_cl - omega_cl)
            case 2:
                y = deriv
            case _:
                raise ValueError('derivative order n must be 1 or 2')

        axs.flat[cnt].plot(omega_cl, y, '.')
        axs.flat[cnt].plot(omega_fit_cl, y, 'r')
        axs.flat[cnt].grid(True)
        axs.flat[cnt].set_title(f'{duration}s')
        cnt += 1
    for ax in axs.flat:
        omega_str = r'$\omega$'
        ax.set_xlabel(f'{omega_str} (deg)')
        match n:
            case 1:
                ylabel_str = r'$ \mathrm{d} \omega / \mathrm{d}t$ - $k_i \eta$ (deg/$\mathrm{s}$)'
            case 2:
                ylabel_str = r'$ \mathrm{d}^2 \omega / \mathrm{d}t^2$ (deg/$\mathrm{s}^2$)'
            case _:
                raise ValueError('derivative order n must be 1 or 2')
        ax.set_ylabel(ylabel_str)
    if title_str is not None:
        fig.suptitle(title_str)
    if save_name is not None:
        fig.savefig(f'{save_name}')
    plt.show()


def get_frac_to_setpt(t, setpt, omega, imask, avg_dt=0.5):
    """ Get the fraction of distance between omega and setpt over an averaging
    window - from end of data. 

    Parameters
    ----------
    t : ndarray
        1D array of time points
    setpt : ndarray
        1D array of set point values
    omega : ndarray
        1D array of state values
    imask : ndarray  
        1D array of bools indicating which values to ignore

    Returns
    -------
    frac_to_setpt : float
        fraction of distance to set point over the averaging window
    """
    mask = ~imask
    t_mask = t[mask]
    setpt_mask = setpt[mask]
    omega_mask = omega[mask]
    avgmask = t_mask > t_mask[-1]  - avg_dt 
    setpt_end = setpt_mask[avgmask].mean()
    omega_end = omega_mask[avgmask].mean()
    frac_to_setpt = 1.0 - np.absolute(setpt_end - omega_end)/setpt_end
    return frac_to_setpt


def get_initial_guess(p_fit_dict):
    """ Get initial guess for pi controller parameters based on p controller fit 
    results. 

    Parameters
    ----------
    p_fit_dict : dict
        dictionary of fit results as returned by fit_p_controller_to_datasets

    Returns
    -------
    guess : list
        initial guess for pi controller parameters [d, kp, ki]

    """
    d_mean = np.array([v['d'] for _,v in p_fit_dict['duration'].items()]).mean()
    kp_mean = np.array([v['kp'] for _,v in p_fit_dict['duration'].items()]).mean()
    guess = [d_mean, kp_mean, 0.0]
    return guess


# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    import pathlib
    from fit_p_controller import fit_p_controller_to_datasets

    data_dir = pathlib.Path('data_mean')

    # Fit p controller and get mean d and kp to use as guess for pi controller fits
    p_fit_dict = fit_p_controller_to_datasets(
            data_dir, 
            omega_min=20.0, 
            tfrac=0.95, 
            disp=False,
            )

    # Fit pi controller
    guess = get_initial_guess(p_fit_dict)
    bounds = ((0.01,100), (0,100), (0.0, 100))

    if 1:
        means_fit_dict = fit_pi_controller_to_datasets(
                data_dir, 
                guess=guess, 
                omega_min=20.0, 
                tfrac=None, 
                disp=True,
                include_ol=True,
                bounds=bounds,
                fit_win_cl=3.0, 
                #fit_win_cl=None, 
                fit_win_ol=None,
                )
        if 1:
            plot_omega_fits(
                    means_fit_dict, 
                    title_str='PI-controller mean fits',
                    save_name='pi_controller_mean_fits.png', 
                    )

        if 1:
            plot_deriv_fits(
                    means_fit_dict, 
                    n = 1, 
                    title_str='PI-controller mean derivative fit', 
                    figsize =(20,12),
                    )

    if 1:
        ensemble_fit_dict = ensemble_fit_pi_controller(
                data_dir, 
                guess=guess, 
                omega_min=20.0, 
                tfrac=None, 
                disp=True,
                include_ol=True,
                bounds=bounds,
                fit_win_cl=3.0, 
                #fit_win_cl=None, 
                fit_win_ol=None,
                )
        if 1:
            plot_omega_fits(
                    ensemble_fit_dict, 
                    title_str='PI-controller ensemble fit',
                    save_name='pi_controller_ensemble_fit.png',
                    )
        if 1:
            plot_deriv_fits(
                    ensemble_fit_dict,
                    n=1, 
                    title_str='PI-controller ensemble derivative fit',
                    figsize =(20,12),
                    )





    
