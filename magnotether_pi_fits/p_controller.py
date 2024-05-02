import copy
import collections
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from . import utility
from . import models


def fit_p_controller_to_datasets(data_dir, guess=None, omega_min=20.0, tfrac=None, disp=False):
    """ Fits proportional controller to all magnotether datasets in given data
    directory.

    Parameters
    ----------
    data_dir : Path or str
        directory containing datasets
    guess : array_like
        initial guess for controller parameters [d, kp]
    omega_min : float  (optional)
        minimum required omega or None. Used as starting point for fit. Start
        of stimulus is used if omega_min is None. 
    tfrac : float or None
        fraction of stimulus duration to fit (all if None)
    disp : bool
        flag specifying whether or not to display results as they are generated

    Returns
    -------
    results_dict : dict
        dictionary of fit results (and original data) for each dataset
        results_dict = {
            'type'     : 'individual', 
            'duration' : {
                duration0 : { 
                    'd'        : d, 
                    'kp'       : kp, 
                    't'        : t, 
                    'omega'    : omega, 
                    'setpt'    : setpt,
                    'disable'  : disable,
                    'omega_fit': omega_fit,
                    'equilib'  : equilib, 
                    },
                duration1 : { ... }, 
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

    results_dict = {'type' : 'individual', 'duration' : {}}
    for duration, data in data_dict.items():
        if disp:
            print(f'duration = {duration}')
        result = fit_p_controller(data, guess=guess, omega_min=omega_min, tfrac=tfrac)
        results_dict['duration'][duration] = result

        if disp:
            print(f'  d       = {result["d"]}')
            print(f'  kp      = {result["kp"]}')
            print(f'  equilib = {result["closed_loop"]["equilib"]}')
            print()
    return results_dict


def fit_p_controller(data, guess=None, omega_min=20.0, tfrac=None):
    """ Fits proportional controller to all magnotether data. 

    Parameters
    ----------
    data : dict 
        directory containing datasets
    guess : array_like
        initial guess for controller parameters [d, kp]
    omega_min : float  (optional)
        minimum required omega or None. Used as starting point for fit. Start
        of stimulus is used if omega_min is None. 
    tfrac : float or None
        fraction of stimulus duration to fit (all if None)

    Returns
    -------
    result : dict
        dictionary of fit results
        result = {
                'd'         : d,
                'kp'        : kp, 
                'closed_loop' : {
                    't'         : t_cl, 
                    'omega'     : omega_cl, 
                    'setpt'     : setpt_cl, 
                    'disable'   : disable_cl, 
                    'omega_fit' : omega_fit_cl,
                    'equilib'   : equilib_cl, 
                    },
                'open_loop' : {
                    't'         : t_ol, 
                    'omega'     : omega_ol, 
                    'setpt'     : setpt_ol, 
                    'disable'   : disable_ol, 
                    'omega_fit' : omega_fit_ol,
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

    # Truncate data to fraction of stimulus duration 
    if tfrac is not None:
        mask = data_cl['t'] < tfrac*duration
        data_cl = utility.apply_mask(data_cl, mask)

    # Extract data arrays for closed loop period
    t_cl = data_cl['t']
    omega_cl = data_cl['omega']
    setpt_cl = data_cl['setpt']
    disable_cl = data_cl['disable']

    # Fit data via least squares using scipy's minimuze. 
    if guess is None:
        guess = [0.0, 1.0]
    res = optimize.minimize(models.p_control_cost, guess, (setpt_cl[0], t_cl, omega_cl)) 
    d, kp = np.absolute(res.x)
    equilib_cl = setpt_cl[0]*kp/(d + kp)
    omega_fit_cl = models.p_control_model(t_cl, d, kp, setpt_cl[0], omega_cl[0])

    # Extract data arrays for open loop period
    t_ol = data_ol['t']
    omega_ol = data_ol['omega']
    setpt_ol = data_ol['setpt']
    disable_ol = data_ol['disable']

    # Get prediction during open loop period
    x0_ol = omega_fit_cl[-1]
    xeq_ol = 0.0
    omega_fit_ol = models.exponential_decay(t_ol, d, x0_ol, xeq_ol)

    result = {
            'd'         : d,
            'kp'        : kp, 
            'closed_loop' : {
                't'         : t_cl, 
                'omega'     : omega_cl, 
                'setpt'     : setpt_cl, 
                'disable'   : disable_cl, 
                'omega_fit' : omega_fit_cl,
                'equilib'   : equilib_cl, 
                },
            'open_loop' : {
                't'         : t_ol, 
                'omega'     : omega_ol, 
                'setpt'     : setpt_ol, 
                'disable'   : disable_ol, 
                'omega_fit' : omega_fit_ol,
                }
            }
    return result

def plot_omega_fits(results_dict, plot_ol=False, save_name=None, figsize=(12,10)): 
    """ Plots the omega fits for the magnotether data. 

    Parameters
    ----------
    results_dict : dict
        dictionary of fit results as returned by fit_p_controller_to_datasets

    plot_ol : bool
        flag indicating whether or not to include openloop data and prediction
        in the plots.

    save_name : str or None
        file name for saving figure.  Not save if None. 

    figsize : tuple
        figure size (w,h)

    """
    fig, ax = plt.subplots(len(results_dict['duration']),1, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0.1)
    plot_cnt = 0

    for duration, result in results_dict['duration'].items(): 
        # Extract closed loop data
        closed_loop = result['closed_loop']
        t_cl = closed_loop['t']
        setpt_cl = closed_loop['setpt']
        omega_cl = closed_loop['omega']
        omega_fit_cl = closed_loop['omega_fit']

        # Extract open loop data
        open_loop = result['open_loop']
        t_ol = open_loop['t']
        setpt_ol = open_loop['setpt']
        omega_ol = open_loop['omega']
        omega_fit_ol = open_loop['omega_fit']

        # Plot closed loop data
        ax[plot_cnt].plot(t_cl, omega_cl, 'o', c='gray', alpha=0.4)
        cl_fit_line,  = ax[plot_cnt].plot(t_cl, omega_fit_cl, 'r', linewidth=2)
        setpt_line, = ax[plot_cnt].plot(t_cl, setpt_cl, 'k', linewidth=2)

        # plot open loop data
        if plot_ol:
            ax[plot_cnt].plot(t_ol, omega_ol, 'o', c='gray', alpha=0.4)
            ol_fit_line, = ax[plot_cnt].plot(t_ol, omega_fit_ol, 'b', linewidth=2)

        ax[plot_cnt].set_ylabel(r'$\omega$')
        ax[plot_cnt].grid(True)
        ax[plot_cnt].set_ylim(0, setpt_cl[0]*1.1)
        if plot_cnt == 0:
            ax[plot_cnt].set_title('P Controller Fit')
            if plot_ol:
                line_list = (setpt_line, cl_fit_line, ol_fit_line)
                legend_list = ('set point', 'fit closed-loop', 'fit open-loop' )
            else:
                line_list = (setpt_line, cl_fit_line)
                legend_list = ('set point', 'fit closed-loop')
            ax[plot_cnt].legend(line_list, legend_list, loc='center right')
        plot_cnt += 1

    ax[len(results_dict)-1].set_xlabel('t (sec)')
    if save_name is not None:
        fig.savefig(save_name)
    plt.show()


def estimate_ki_from_openloop(results_dict, end_win_cl=0.5,  end_win_ol=10.0, disp=False):
    """ Estimates the integral ki from the p controller fits and open-loop data 

    Parameters
    ----------
    results_dict : dict
        dictionary of fit results as returned by fit_p_controller_to_datasets

    end_win_cl : float
        averaging window (s) used to find final value of omega in closed-loop data section. 

    end_win_ol : float
        averaging window (s) used to find final value of omega in open-loop data section. 

    """
    if disp:
        print()

    ierr = collections.OrderedDict()
    omega_end_cl = collections.OrderedDict()
    omega_end_ol = collections.OrderedDict()
    ki_estimate  = collections.OrderedDict()

    for duration, result in results_dict['duration'].items(): 
        if disp:
            print(f'duration: {duration}')

        # Extract closed loop data
        closed_loop = result['closed_loop']
        t_cl = closed_loop['t']
        setpt_cl = closed_loop['setpt']
        omega_cl = closed_loop['omega']
        omega_fit_cl = closed_loop['omega_fit']

        # Extract open loop data
        open_loop = result['open_loop']
        t_ol = open_loop['t']
        setpt_ol = open_loop['setpt']
        omega_ol = open_loop['omega']
        omega_fit_ol = open_loop['omega_fit']

        # Get integral error for both raw data and model fit
        dt = t_cl[1] - t_cl[0]
        ierr_data  = np.trapz(setpt_cl - omega_cl, dx=dt) 
        ierr_model = np.trapz(setpt_cl - omega_fit_cl, dx=dt)
        ierr[duration] = { 
                'data'  : ierr_data, 
                'model' : ierr_model, 
                }

        # Get final value for omega during closed loop period 
        mask = t_cl > (t_cl[-1] - end_win_cl)
        omega_end_cl_data  = omega_cl[mask].mean() 
        omega_end_cl_model = omega_cl[-1] 
        omega_end_cl[duration] = {
                'data'  : omega_end_cl_data, 
                'model' : omega_end_cl_model, 
                }

        # Get final value for omega during open loop period
        mask = t_ol > (t_ol[-1] - end_win_ol)
        omega_end_ol_data = omega_ol[mask].mean()
        omega_end_ol[duration] = omega_end_ol_data

        # Get damping coeff, open-loop duration and exponential decay for ki estimates
        d = result['d']
        duration_ol = t_ol[-1] - t_ol[0]
        exp_decay = np.exp(-d*duration_ol)

        ki_data = (d/ierr_data)
        ki_data *= omega_end_ol_data - omega_end_cl_data*exp_decay
        ki_data /= 1.0 - exp_decay

        ki_model = (d/ierr_model)
        ki_model *= omega_end_ol_data - omega_end_cl_model*exp_decay
        ki_model /= 1.0 - exp_decay

        ki_estimate[duration] = {
                'data'  : ki_data, 
                'model' : ki_model, 
                }

        if disp:
            print(f'  ierr')
            print(f'    data:  {ierr_data}')
            print(f'    model: {ierr_model}')
            print()
            print(f'  omega_end_cl')
            print(f'    data:  {omega_end_cl_data}')
            print(f'    model: {omega_end_cl_model}')
            print()
            print(f'  omega_end_ol: {omega_end_ol_data}')
            print()
            print(f'  ki')
            print(f'    data:  {ki_data}')
            print(f'    model: {ki_model}')
            print()

    ki_results = { 
            'ierr'         : ierr, 
            'omega_end_cl' : omega_end_cl, 
            'omega_end_ol' : omega_end_ol, 
            'ki_estimate'  : ki_estimate, 
            }

    return ki_results


def plot_ki_estimates(ki_results):
    """
    Plots the integral gain estimates and the various data used to make the estimate.

    Parameters
    ----------
    ki_results : dict
        dictionary of results from estimating the integral gains as returned by
        estimate_ki_from_openloop

    Plots
    -----

    * integral error during closed-loop vs stimulus duration
    * omega at end of closed-loop period vs stimulus duration
    * omega at end of open-loop perid vs stimulus duration
    * estimated integral gain  vs stimulus duration

    """

    # Extract data to arrays
    duration = np.array([k for k,_ in ki_results['ierr'].items()])
    ierr_data = np.array([v['data'] for _,v in ki_results['ierr'].items()])
    ierr_model = np.array([v['model'] for _,v in ki_results['ierr'].items()])
    omega_end_cl_data = np.array([v['data'] for _,v in ki_results['omega_end_cl'].items()])
    omega_end_cl_model = np.array([v['model'] for _,v in ki_results['omega_end_cl'].items()])
    omega_end_ol_data = np.array([v for _,v in ki_results['omega_end_ol'].items()])
    ki_data = np.array([v['data'] for _,v in ki_results['ki_estimate'].items()])
    ki_model = np.array([v['model'] for _,v in ki_results['ki_estimate'].items()])

    fig, ax = plt.subplots(1,1)
    data_line, = ax.plot(duration, ierr_data, '-ob')
    model_line, = ax.plot(duration, ierr_model, '-og')
    ax.grid(True)
    ax.set_xlabel('duration (s)')
    ax.set_ylabel('integral error (deg)')
    ax.legend((data_line, model_line), ('data', 'fit'), loc='upper left')
    ax.set_title('Integral Error')

    fig, ax = plt.subplots(1,1)
    data_line, = ax.plot(duration, omega_end_cl_data, '-ob')
    model_line, = ax.plot(duration, omega_end_cl_model, '-og')
    ax.grid(True)
    ax.set_xlabel('duration (s)')
    ax.set_ylabel('omega (deg/s)')
    ax.legend((data_line, model_line), ('data', 'fit'), loc='upper left')
    ax.set_title('Omega at end of closed-loop period')

    fig, ax = plt.subplots(1,1)
    data_line, = ax.plot(duration, omega_end_ol_data, '-ob')
    ax.grid(True)
    ax.set_xlabel('duration (s)')
    ax.set_ylabel('omega (deg/s)')
    ax.set_title('Omega at end of open-loop period')

    fig, ax = plt.subplots(1,1)
    data_line, = ax.plot(duration, ki_data, '-ob')
    model_line, = ax.plot(duration, ki_model, '-og')
    ax.grid(True)
    ax.set_xlabel('duration (s)')
    ax.set_ylabel('gain (1/s)')
    ax.set_title('Estimated integral gain')
    plt.show()


def plot_deriv_fits(results_dict, title_str=None, figsize=(20,12), save_name=None):
    """ Plots derivative of fits of p control model to magnotether data.

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
    ---------

    """
    num_duration = len(results_dict['duration'])

    ngrid = int(np.ceil(np.sqrt(num_duration)))

    fig, axs = plt.subplots(ngrid, ngrid, figsize=figsize, sharex=True)
    cnt = 0
    for duration, trial_data in results_dict['duration'].items():

        if results_dict['type'] == 'individual':
            d = trial_data['d']
            kp = trial_data['kp']
        else:
            d = results_dict['d']
            kp = results_dict['kp']

        data_cl = trial_data['closed_loop']
        data_ol = trial_data['open_loop']

        t_cl = data_cl['t']
        setpt_cl = data_cl['setpt']
        omega_cl = data_cl['omega']
        omega_fit_cl = data_cl['omega_fit']
        deriv =  models.p_control_derivs(t_cl, d, kp, setpt_cl[0], omega_cl[0])

        axs.flat[cnt].plot(omega_cl, deriv, '.')
        axs.flat[cnt].plot(omega_fit_cl, deriv, 'r')
        axs.flat[cnt].grid(True)
        axs.flat[cnt].set_title(f'{duration}s')
        cnt += 1
    for ax in axs.flat:
        omega_str = r'$\omega$'
        ax.set_xlabel(f'{omega_str} (deg)')
        ylabel_str = r'$ \mathrm{d} \omega / \mathrm{d}t$  (deg/$\mathrm{s}$)'
        ax.set_ylabel(ylabel_str)
    if title_str is not None:
        fig.suptitle(title_str)
    if save_name is not None:
        fig.savefig(f'{save_name}')
    plt.show()



