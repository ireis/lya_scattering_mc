import numpy
import fit_binned_nof_photons_v3 as fbp
from common_calcs import *
from importlib import reload
import pandas




def fit_all_gammas():
    for za in za_grid:
        za_df = pandas.read_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(za))
        x = za_df['r_shell'].values
        y = za_df['m'].values
        fit_loc = numpy.where((x < 400) & (y < 5) & (y > 1))[0]
        z = numpy.polyfit(x[fit_loc], y[fit_loc], 3)
        #p = numpy.poly1d(z)
        numpy.save('fitting_results_v3/gammas/gamma_poly_za_{}.npy'.format(za), z)

def fit_all_rmin():
    for za in za_grid:
        za_df = pandas.read_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(za))
        x = za_df['r_shell'].values
        m_ = za_df['m'].values
        try:
            n_dists = za_df['nof_dists'].values
            n_dists_norm = 1000000
            delta_log_c = numpy.log10(n_dists_norm/n_dists)
            delta_lr_min = -delta_log_c/m_
        except:
            delta_log_c = 0
            delta_lr_min = 0
        y = 10**(za_df['lr_min'].values + delta_lr_min)
        fit_loc = numpy.where((x < 400) & (m_ < 5) & (m_ > 1)  & (y < x))[0]
        z = numpy.polyfit(x[fit_loc], y[fit_loc], 2)
        numpy.save('fitting_results_v3/rmins/rmin_poly_za_{}.npy'.format(za), z)




def fit_all_rmax():
    for za in za_grid:
        za_df = pandas.read_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(za))
        x = za_df['r_shell'].values
        y = 10**(za_df['lr_max'].values)
        m_ = za_df['m'].values
        fit_loc = numpy.where((x < 400) & (m_ < 5) & (m_ > 1)  & (y < x))[0]
        z = numpy.polyfit(x[fit_loc], y[fit_loc], 2)
        numpy.save('fitting_results_v3/rmaxs/rmax_poly_za_{}.npy'.format(za), z)




def fit_all_rmax():
    for za in za_grid:
        za_df = pandas.read_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(za))
        x = za_df['r_shell'].values
        y = 10**(za_df['lr_max'].values)
        m_ = za_df['m'].values
        fit_loc = numpy.where((x < 400) & (m_ < 5) & (m_ > 1)  & (y < x))[0]
        z = numpy.polyfit(x[fit_loc], y[fit_loc], 2)
        numpy.save('fitting_results_v3/rmaxs/rmax_poly_za_{}.npy'.format(za), z)



def fit_all_alpha():
    for za in za_grid:
        za_df = pandas.read_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(za))
        x = za_df['r_shell'].values
        y = za_df['alpha'].values
        rmx =  10**(za_df['lr_max'].values)
        m_ = za_df['m'].values
        fit_loc = numpy.where((x < 400) & (m_ < 5) & (m_ > 1)  & (rmx < x))[0]
        z = numpy.polyfit(x[fit_loc], y[fit_loc], 2)
        numpy.save('fitting_results_v3/alphas/alpha_poly_za_{}.npy'.format(za), z)


def fit_all_beta():
    for za in za_grid:
        za_df = pandas.read_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(za))
        x = za_df['r_shell'].values
        y = za_df['beta'].values
        rmx =  10**(za_df['lr_max'].values)
        m_ = za_df['m'].values
        fit_loc = numpy.where((x < 400) & (m_ < 5) & (m_ > 1)  & (rmx < x))[0]
        z = numpy.polyfit(x[fit_loc], y[fit_loc], 2)
        numpy.save('fitting_results_v3/betas/beta_poly_za_{}.npy'.format(za), z)


def fit_all_bst():
    for za in za_grid:
        za_df = pandas.read_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(za))
        x = za_df['r_shell'].values
        y = za_df['bst'].values
        rmx =  10**(za_df['lr_max'].values)
        m_ = za_df['m'].values
        fit_loc = numpy.where((x < 400) & (m_ < 5) & (m_ > 1)  & (rmx < x))[0]
        z = numpy.polyfit(x[fit_loc], y[fit_loc], 2)
        numpy.save('fitting_results_v3/bsts/bst_poly_za_{}.npy'.format(za), z)



if __name__ == '__main__':

    za_grid = numpy.arange(5,41)

    Rshell_grid, Rmin, Rmax, nof_shells = get_Rshell_grid()

    mc_results_path = '/Volumes/Backup/Cosmo/distance_distribution/dense_z_source_1000_nu_bins_20000_r/'

    fit_all_bst()
    fit_all_beta()
    fit_all_alpha()
    fit_all_rmax()
    fit_all_rmax()
    fit_all_rmin()
    fit_all_gammas()
