import pathlib
from magnotether_pi_fits.p_controller import fit_p_controller_to_datasets
from magnotether_pi_fits.p_controller import estimate_ki_from_openloop
from magnotether_pi_fits.p_controller import plot_omega_fits
from magnotether_pi_fits.p_controller import plot_deriv_fits
from magnotether_pi_fits.p_controller import plot_ki_estimates 

data_dir = pathlib.Path('data')

fit_results = fit_p_controller_to_datasets(
        data_dir, 
        guess=[0.0, 3.0], 
        omega_min=20.0, 
        tfrac=None, 
        disp=False,
        )

if 1:
    plot_omega_fits(fit_results, plot_ol=True)

if 1:
    plot_deriv_fits(
            fit_results, 
            title_str='P-controller mean derivative fits (closed-loop period)'
            )

if 1:
    ki_results = estimate_ki_from_openloop(
            fit_results, 
            end_win_cl = 0.5, 
            end_win_ol = 10.0,
            disp=False,
            )
    plot_ki_estimates(ki_results)

