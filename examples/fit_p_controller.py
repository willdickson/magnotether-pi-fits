import pathlib
from magnotether_pi_fits.p_controller import fit_p_controller_to_datasets
from magnotether_pi_fits.p_controller import estimate_ki_from_openloop
from magnotether_pi_fits.p_controller import plot_omega_fits
from magnotether_pi_fits.p_controller import plot_deriv_fits
from magnotether_pi_fits.p_controller import plot_ki_estimates 
from magnotether_pi_fits.p_controller import plot_p_and_pi_openloop

data_dir = pathlib.Path('data/mean')

p_fit_dict = fit_p_controller_to_datasets(
        data_dir, 
        guess=[0.0, 3.0], 
        omega_min=20.0, 
        tfrac=None, 
        disp=False,
        )

if 1:
    plot_omega_fits(p_fit_dict, plot_ol=True)

if 1:
    plot_deriv_fits(
            p_fit_dict, 
            title_str='P-controller mean derivative fits (closed-loop period)'
            )

if 1:
    ki_results = estimate_ki_from_openloop(
            p_fit_dict, 
            end_win_cl = 0.5, 
            end_win_ol = 10.0,
            disp=False,
            )
    if 1:
        plot_ki_estimates(ki_results)

    if 1:
        plot_p_and_pi_openloop(p_fit_dict, ki_results)

