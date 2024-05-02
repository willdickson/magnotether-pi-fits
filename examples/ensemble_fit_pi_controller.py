import pathlib
from magnotether_pi_fits.p_controller import fit_p_controller_to_datasets
from magnotether_pi_fits.pi_controller import fit_pi_controller_to_datasets
from magnotether_pi_fits.pi_controller import ensemble_fit_pi_controller
from magnotether_pi_fits.pi_controller import get_initial_guess
from magnotether_pi_fits.pi_controller import plot_omega_fits
from magnotether_pi_fits.pi_controller import plot_deriv_fits

data_dir = pathlib.Path('data/mean')

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

pi_fit_dict = ensemble_fit_pi_controller(
        data_dir, 
        guess=guess, 
        omega_min=20.0, 
        tfrac=None, 
        disp=True,
        include_ol=True,
        bounds=bounds,
        fit_win_cl=3.0, 
        fit_win_ol=None,
        )
if 1:
    plot_omega_fits(
            pi_fit_dict, 
            title_str='PI-controller ensemble fit',
            save_name='pi_controller_ensemble_fit.png',
            )
if 1:
    plot_deriv_fits(
            pi_fit_dict,
            n=1, 
            title_str='PI-controller ensemble derivative fit',
            figsize =(20,12),
            )

